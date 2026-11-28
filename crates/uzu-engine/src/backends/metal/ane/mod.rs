//! Apple Neural Engine (ANE) offload of prefill, on top of the `ane` crate.
//!
//! One model and one prefill at a time per context, since the surfaces and events are the context's;
//! an ANE failure errors out the prefill in flight and retires the ANE.
//!
//! The FFN program is adapted from Splash (<https://github.com/incoai/splash>, commit 031a25a),
//! Apache License 2.0: `runtime/ops/AneFfn.cpp`. Not affiliated with Splash.

mod program;

use std::{
    ffi::c_void,
    ptr,
    range::Range,
    sync::{Arc, atomic::Ordering},
};

use ane::w8a8::{
    Host, Launch, Layer, Region, Residency, Rotation, Session, TOKENS, W4, W4Layout, WeightCodeLayout, WeightLayer,
    WeightSource,
};
use metal::{MTLBuffer, MTLDevice, MTLResidencySet};
use objc2::{rc::Retained, runtime::ProtocolObject};
use program::Offloaded;

use crate::{
    backends::{
        common::{
            BufferRef, CommandBufferEncoding, CpuBuffer,
            gpu_types::quantization::QuantizationMode,
            kernel::matmul::{MatmulB, QuantParamsLayout, QuantizedCorrection},
        },
        metal::{
            Metal,
            buffer::MetalBufferExt,
            command_buffer::{Access, MetalCommandBufferEncoding},
            context::MetalContext,
            error::MetalError,
            gpu_types::gemm::GemmBLayout,
        },
    },
    data_type::DataType,
    encodable_block::{
        linear::{AneWeightLayer, Linear},
        mlp::{DenseMlp, Mlp},
    },
};

/// Smaller batches stay on the GPU (the ANE's fixed cost dominates), larger ones are not compiled for.
const MIN_TOKENS: u32 = 256;
const MAX_TOKENS: u32 = *TOKENS.end();

/// The ANE of one Metal context: programs built on demand per MLP shape and linear output width.
pub struct Ane {
    /// Whether the DeltaNet and attention projections run on the ANE, not only the MLPs.
    projections: bool,
    session: Session,
}

impl Ane {
    pub fn new(gpu_core_count: u32) -> Self {
        Self {
            projections: gpu_core_count <= 10,
            session: Session::default(),
        }
    }

    pub fn disable(&self) {
        self.session.retire();
    }

    /// Runs `mlp` on the ANE if it can, otherwise as built.
    pub fn wrap_mlp(
        &self,
        context: &MetalContext,
        mlp: DenseMlp<Metal>,
    ) -> Box<dyn Mlp<Metal>> {
        let mlp = Arc::new(mlp);
        if let Some((up, down)) = mlp.ane_weights()
            && let Some(layer) = self.offload(|| {
                let (Some(up), Some(down)) = (weight_layer(&up), weight_layer(&down)) else {
                    return Ok(None);
                };
                // SAFETY: the immutable regions are owned by mlp on this context's device.
                let Some(layer) = (unsafe { self.session.bind_ffn(&up, &down, mlp.clone(), || new_host(context)) })?
                else {
                    return Ok(None);
                };
                Ok(Some(layer))
            })
        {
            return Box::new(Offloaded {
                inner: mlp,
                layer,
            });
        }
        Box::new(Arc::try_unwrap(mlp).unwrap_or_else(|_| unreachable!("unbound MLP has one owner")))
    }

    /// Runs `linear` on the ANE if the placement rule and the layer allow it, otherwise as built.
    pub fn wrap_linear(
        &self,
        context: &MetalContext,
        linear: Box<dyn Linear<Metal>>,
        data_type: DataType,
    ) -> Box<dyn Linear<Metal>> {
        let linear = Arc::new(linear);
        if self.projections
            && data_type == DataType::BF16
            && let Some(weights) = linear.ane_weights(None)
            && let Some(layer) = self.offload(|| {
                let Some(weights) = weight_layer(&weights) else {
                    return Ok(None);
                };
                // SAFETY: the immutable regions are owned by linear on this context's device.
                let Some(layer) =
                    (unsafe { self.session.bind_linear(&weights, linear.clone(), || new_host(context)) })?
                else {
                    return Ok(None);
                };
                Ok(Some(layer))
            })
        {
            return Box::new(Offloaded {
                inner: linear,
                layer,
            });
        }
        Arc::try_unwrap(linear).unwrap_or_else(|_| unreachable!("unbound linear has one owner"))
    }

    /// Builds or retrieves a bound layer; failures stay local to that layer unless the ANE retired.
    fn offload(
        &self,
        bind: impl FnOnce() -> Result<Option<Layer>, ane::Error>,
    ) -> Option<Layer> {
        match bind() {
            Ok(layer) => layer,
            Err(error) => {
                if self.session.is_retired() {
                    eprintln!("uzu: the Neural Engine is unavailable, running on the GPU: {error}");
                } else {
                    eprintln!("uzu: ANE layer setup failed, the layer stays on the GPU: {error}");
                }
                None
            },
        }
    }

    /// The chunk lengths of a prefill whose chunk pairs overlap layer by layer, if an MLP program
    /// has been built and the ANE remains healthy.
    pub fn prefill_chunk_lengths(
        &self,
        prompt_length: usize,
    ) -> Option<Vec<usize>> {
        if self.session.has_ffn_program() {
            Some(pipelined_chunk_lengths(prompt_length))
        } else {
            None
        }
    }
}

/// The Neural Engine's host on `context`: its device, with the buffers the programs create joining
/// the context's residency set for as long as they live.
fn new_host(context: &MetalContext) -> Result<Host, ane::Error> {
    let (add_set, remove_set) = (context.residency_set.clone(), context.residency_set.clone());
    let (device, peak_memory_usage) = (context.device.clone(), context.peak_memory_usage.clone());
    let residency = Residency {
        add: Box::new(move |buffer| {
            // SAFETY: the pointer is the `id<MTLBuffer>` the program just created.
            let buffer = unsafe { &*buffer.cast::<ProtocolObject<dyn MTLBuffer>>() };
            let residency_set = add_set.lock();
            residency_set.add_allocation(buffer.as_ref());
            residency_set.commit();
            residency_set.request_residency();
            drop(residency_set);
            peak_memory_usage.fetch_max(device.current_allocated_size(), Ordering::Relaxed);
        }),
        remove: Box::new(move |buffer| {
            // SAFETY: the pointer is the `id<MTLBuffer>` the program is releasing.
            let buffer = unsafe { &*buffer.cast::<ProtocolObject<dyn MTLBuffer>>() };
            let residency_set = remove_set.lock();
            residency_set.remove_allocation(buffer.as_ref());
            residency_set.commit();
        }),
    };
    // SAFETY: the device is live.
    unsafe { Host::new(ptr::from_ref(&*context.device).cast_mut().cast::<c_void>(), residency) }
}

/// Chunks run through the layers in overlapped pairs of at most `2 * MAX_TOKENS` tokens: the prompt is
/// cut into an even number of near-equal chunks. A short prompt stays one chunk.
fn pipelined_chunk_lengths(prompt_length: usize) -> Vec<usize> {
    if prompt_length < 2 * MIN_TOKENS as usize {
        return vec![prompt_length];
    }
    let count = 2 * prompt_length.div_ceil(2 * MAX_TOKENS as usize);
    (0..count).map(|index| prompt_length / count + usize::from(index < prompt_length % count)).collect()
}

fn region(buffer: impl BufferRef<Backend = Metal>) -> Region {
    let (buffer, range) = buffer.parts();
    let range = buffer.gpu_address_subrange(range);
    Region {
        address: range.start,
        len: (range.end - range.start) as usize,
    }
}

fn weight_layer<'a>(layer: &AneWeightLayer<'a, Metal>) -> Option<WeightLayer<'a>> {
    let MatmulB::Quantized(quantized) = layer.matrix.matmul_b() else {
        return None;
    };
    if quantized.params.layout() != QuantParamsLayout::GroupOutput
        || quantized.params.scale_strides().output_stride != 1
    {
        return None;
    }
    let scale_stride = quantized.params.scale_strides().group_stride;
    let scale_stride = if scale_stride == 0 {
        quantized.params.scale_shape()[1]
    } else {
        scale_stride
    };
    if scale_stride < layer.rows {
        return None;
    }
    let codes = region(quantized.codes);
    let scales = region(quantized.scales);
    let source = match (quantized.mode, quantized.correction) {
        (QuantizationMode::U4, QuantizedCorrection::ZeroPoints(zero_points)) => {
            if quantized.group_size != 32 || quantized.signed_codes {
                return None;
            }
            let zero_point_strides = quantized.params.zero_point_strides(quantized.mode);
            let zero_point_stride = if zero_point_strides.group_stride == 0 {
                quantized.params.zero_point_shape(quantized.mode)[1].checked_mul(2)?
            } else {
                zero_point_strides.group_stride
            };
            if zero_point_strides.output_stride != 1 || zero_point_stride != scale_stride {
                return None;
            }
            WeightSource::W4(W4 {
                codes,
                scales,
                zero_points: region(zero_points),
                scale_stride,
                layout: match quantized.weights_layout {
                    GemmBLayout::RowMajor => W4Layout::RowMajor,
                    GemmBLayout::InterleavedRows => W4Layout::InterleavedRows,
                    GemmBLayout::MxuTiled => W4Layout::MxuTiled,
                },
            })
        },
        (QuantizationMode::U8, QuantizedCorrection::Symmetric) => {
            let layout = code_layout(quantized.weights_layout)?;
            WeightSource::SymmetricU8 {
                codes,
                scales,
                scale_stride,
                group_size: quantized.group_size,
                layout,
                signed_codes: quantized.signed_codes,
            }
        },
        (QuantizationMode::U2, QuantizedCorrection::Biases(biases)) => {
            let layout = code_layout(quantized.weights_layout)?;
            WeightSource::MlxU2ScaleBias {
                codes,
                scales,
                biases: region(biases),
                scale_stride,
                group_size: quantized.group_size,
                layout,
                signed_codes: quantized.signed_codes,
            }
        },
        _ => return None,
    };
    let input_rotation = Rotation {
        block_size: layer.input_hadamard_size,
        signs: layer.input_signs.as_slice(),
    };
    let output_rotation = match layer.output_signs {
        Some(signs) => Some(Rotation {
            block_size: layer.output_hadamard_size?,
            signs: signs.as_slice(),
        }),
        None => None,
    };
    Some(WeightLayer {
        source,
        rows: layer.rows,
        columns: layer.columns,
        input_rotation,
        output_rotation,
    })
}

fn code_layout(layout: GemmBLayout) -> Option<WeightCodeLayout> {
    match layout {
        GemmBLayout::RowMajor => Some(WeightCodeLayout::RowMajor),
        GemmBLayout::MxuTiled => Some(WeightCodeLayout::MxuTiled),
        GemmBLayout::InterleavedRows => None,
    }
}

/// Encodes `launch` on `command_buffer`, declaring its accesses for the buffer's barrier tracking.
fn launch(
    command_buffer: &mut MetalCommandBufferEncoding,
    launch: &Launch,
) -> Result<(), MetalError> {
    let command_buffer = &mut *command_buffer.debug_group(launch.name);
    let access = |region: &Region, write: bool| Access {
        range: Range::from(region.address..region.address + region.len as u64),
        write,
    };
    let mut accesses: Vec<_> = launch.buffers.iter().map(|(region, write)| access(region, *write)).collect();
    // A kernel without constants never reads the argument, so there is nothing to allocate.
    let constants_address = if launch.constants.is_empty() {
        0
    } else {
        let constants = region(&command_buffer.allocate_constant_from_slice(&launch.constants)?);
        accesses.push(access(&constants, false));
        constants.address
    };
    command_buffer.access(&accesses);
    // SAFETY: the encoder and its argument table are live, and the memory the launch binds is resident.
    unsafe {
        launch.encode(
            Retained::as_ptr(&command_buffer.compute_encoder).cast_mut().cast(),
            Retained::as_ptr(&command_buffer.argument_table).cast_mut().cast(),
            constants_address,
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../../unit/backends/metal/ane_chunking_test.rs"]
mod chunking_tests;
