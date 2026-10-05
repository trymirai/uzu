//! Quantized matmul for a few activation rows on WMMA: the native kernel `kernel/native/qmv_wmma.clcpp`
//! decodes each 4-bit weight once into a WMMA operand and multiplies 16 activation rows with one
//! `v_wmma_f32_16x16x16_bf16`, so verifying a speculation tree of up to 16 tokens reads the weights once, as a
//! decode step does. The output random Hadamard transform (and the bias after it) runs in the kernel's
//! epilogue instead of a separate ActivationTransform pass. The MSL GEMV keeps M = 1 and everything this kernel
//! does not take.
//!
//! With int8 activations (uzu's A8 path, `MatmulA::Int8Symmetric`) the kernel multiplies with
//! `v_wmma_i32_16x16x16_iu8`; the engine then quantizes activations in the input transform it runs anyway
//! (`select_activation_quantization` / `select_activation_format`). `UZU_AMDGPU_A8=0` keeps bf16 activations.

use crate::{
    backends::{
        amdgpu::{
            Amdgpu,
            buffer::AmdgpuBufferExt,
            kernel::{Kernarg, native},
        },
        common::{
            Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::{
                ActivationQuantization,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{
                    Int8CodeLayout, MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulShape, QuantParamsLayout,
                },
            },
        },
    },
    data_type::DataType,
};

/// Waves of a workgroup, each over its share of K (`QMV_WAVES` in the kernel).
const WAVES: u32 = 8;
/// The kernel reads K in 128-wide blocks.
const K_MULTIPLE: u32 = 128;
const ROWS_PER_GROUP: u32 = 16;

const FLAG_SIGNED_CODES: u32 = 1;
const FLAG_SCALE: u32 = 2;
const FLAG_BIAS: u32 = 4;
const FLAG_RHT: u32 = 8;
/// The output random Hadamard transform works on blocks of 32 columns: the 32-column tile of one workgroup.
const RHT_BLOCK: u32 = 32;

fn env_u32(
    name: &str,
    default: u32,
) -> u32 {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

/// Batch sizes routed here: from `UZU_AMDGPU_QMV_WMMA_MIN_M` (default 2) to `UZU_AMDGPU_QMV_WMMA_MAX_M`
/// (default: up to where GEMM takes over).
pub fn m_range() -> (u32, u32) {
    static RANGE: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
    *RANGE.get_or_init(|| (env_u32("UZU_AMDGPU_QMV_WMMA_MIN_M", 2), env_u32("UZU_AMDGPU_QMV_WMMA_MAX_M", u32::MAX)))
}

/// Output columns per wave: 16 by default; 32 measured no faster (`UZU_AMDGPU_QMV_WMMA_TILES` = 2 overrides).
fn tiles() -> u32 {
    static OVERRIDE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    let tiles = *OVERRIDE.get_or_init(|| {
        std::env::var("UZU_AMDGPU_QMV_WMMA_TILES")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|t| matches!(t, 1 | 2))
    });
    tiles.unwrap_or(1)
}

/// Int8 activations for the shapes this kernel takes (`UZU_AMDGPU_A8=0` disables them).
pub fn a8_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("UZU_AMDGPU_A8").map_or(true, |value| value != "0"))
}

/// The activation quantization the int8 path reads: 128-wide scale groups, GroupedByNibble codes, no group
/// sums (zero points come off in the weight bytes). `candidate` is the int8 shape of a linear layer.
pub fn activation_quantization(
    candidate: &MatmulShape,
    data_types: [DataType; 3],
) -> Option<ActivationQuantization> {
    if !a8_enabled() || candidate.params_layout != Some(QuantParamsLayout::GroupOutput) {
        return None;
    }
    let a8_shape = MatmulShape {
        a_full_precision: false,
        ..*candidate
    };
    if !supports(&a8_shape, data_types) {
        return None;
    }
    ActivationQuantization::new(
        ACTIVATION_SCALE_GROUP_SIZE,
        candidate.b_group_size?,
        false,
        Int8CodeLayout::GroupedByNibble,
    )
}

fn address(buffer: impl BufferRef<Backend = Amdgpu>) -> u64 {
    let (buffer, range) = buffer.parts();
    buffer.device_address() + range.start as u64
}

pub fn supports(
    shape: &MatmulShape,
    data_types: [DataType; 3],
) -> bool {
    // int8 activations: zero-point and symmetric weights (the zero point comes off in the weight bytes)
    let prologue_supported = match shape.b_prologue {
        GemmBPrologueKind::ScaleZeroPointDequant | GemmBPrologueKind::ScaleSymmetricDequant => true,
        GemmBPrologueKind::ScaleBiasDequant => shape.a_full_precision,
        GemmBPrologueKind::FullPrecision => false,
    };
    !native::QMV_WMMA.is_empty()
        && prologue_supported
        && shape.b_bits == Some(4)
        && matches!(shape.b_group_size, Some(32 | 64))
        && shape.b_transpose
        && shape.b_leading_dimension.is_none()
        && (shape.a_full_precision || a8_enabled())
        && !shape.gathered
        && data_types.iter().all(|&data_type| data_type == DataType::BF16)
        && shape.k.is_multiple_of(K_MULTIPLE)
        && shape.n > 0
        && !shape.d_transform.intersects(GemmDTransform::ACCUMULATE | GemmDTransform::SOFT_CAP)
        && (!shape.d_transform.contains(GemmDTransform::RHT) || shape.n.is_multiple_of(RHT_BLOCK))
}

/// The kernel reads activation and code rows as 16-byte vectors.
pub fn aligned(
    arguments: &MatmulArguments<
        '_,
        Amdgpu,
        impl BufferRef<Backend = Amdgpu>,
        impl BufferRef<Backend = Amdgpu>,
        impl BufferMut<Backend = Amdgpu>,
        impl BufferRef<Backend = Amdgpu>,
    >
) -> bool {
    let a = match &arguments.a {
        MatmulA::FullPrecision {
            values,
            offset,
        } => address(*values) + *offset as u64,
        MatmulA::Int8Symmetric {
            values,
            ..
        } => address(*values),
    };
    let MatmulB::Quantized(quantized) = &arguments.b else {
        return false;
    };
    a.is_multiple_of(16) && address(quantized.codes).is_multiple_of(16)
}

pub fn encode(
    arguments: MatmulArguments<
        '_,
        Amdgpu,
        impl BufferRef<Backend = Amdgpu>,
        impl BufferRef<Backend = Amdgpu>,
        impl BufferMut<Backend = Amdgpu>,
        impl BufferRef<Backend = Amdgpu>,
    >,
    command_buffer: &mut <<Amdgpu as Backend>::CommandBuffer as CommandBuffer>::Encoding,
) -> Result<(), MatmulError<Amdgpu>> {
    let MatmulArguments {
        a,
        b,
        d,
        d_transform,
        m,
        n,
        k,
        ..
    } = arguments;
    // (activation address, activation scales address, int8)
    let (a_address, a_scales, a8) = match a {
        MatmulA::FullPrecision {
            values,
            offset,
        } => (address(values) + offset as u64, 0, false),
        MatmulA::Int8Symmetric {
            values,
            scales,
            scale_group_size,
            code_layout,
            ..
        } => {
            if scale_group_size != ACTIVATION_SCALE_GROUP_SIZE || code_layout != Int8CodeLayout::GroupedByNibble {
                return Err(MatmulError::IncompatibleA {
                    path: "QmvWmma",
                    reason: "int8 activations need 128-wide scale groups and the GroupedByNibble layout",
                });
            }
            (address(values), address(scales), true)
        },
    };
    let MatmulB::Quantized(quantized) = b else {
        return Err(MatmulError::IncompatibleA {
            path: "QmvWmma",
            reason: "full-precision weights",
        });
    };

    // The output transform runs in the epilogue: the 32-column tile holds whole Hadamard blocks, and the bias
    // follows the transform as in the MSL GEMV epilogue.
    let rht_factors = d_transform.rht_factors;
    let bias = d_transform.bias;
    let prologue = match quantized.correction.zero_points().is_some() {
        true => "zero_point",
        false if quantized.correction.biases().is_some() => "scale_bias",
        false => "symmetric",
    };
    let tiles = if rht_factors.is_some() {
        RHT_BLOCK / ROWS_PER_GROUP
    } else {
        tiles()
    };
    let entry = format!(
        "qmv_wmma_{}w4_g{}_{prologue}_t{tiles}",
        if a8 {
            "a8_"
        } else {
            ""
        },
        quantized.group_size
    );
    let function = command_buffer.context().function(&[native::QMV_WMMA], &entry).map_err(MatmulError::BackendError)?;

    let mut flags = 0;
    if quantized.signed_codes {
        flags |= FLAG_SIGNED_CODES;
    }
    if d_transform.ab_scale != 1.0 {
        flags |= FLAG_SCALE;
    }
    if bias.is_some() {
        flags |= FLAG_BIAS;
    }
    if rht_factors.is_some() {
        flags |= FLAG_RHT;
    }
    let scale_strides = quantized.params.scale_strides();
    let zero_point_strides = quantized.zero_point_strides();

    let mut kernarg = Kernarg::new();
    kernarg.push_address(address(quantized.codes));
    kernarg.push_address(address(quantized.scales));
    kernarg.push_address(quantized.correction.zero_points().map_or(0, |&buffer| address(buffer)));
    kernarg.push_address(quantized.correction.biases().map_or(0, |&buffer| address(buffer)));
    kernarg.push_address(a_address);
    kernarg.push_address(a_scales);
    let (d_buffer, d_range) = d.parts();
    kernarg.push_address(d_buffer.device_address() + d_range.start as u64);
    kernarg.push_address(bias.map_or(0, address));
    kernarg.push_address(rht_factors.map_or(0, address));
    for value in [
        m,
        n,
        k,
        scale_strides.output_stride,
        scale_strides.group_stride,
        zero_point_strides.output_stride,
        zero_point_strides.group_stride,
        d_transform.ab_scale.to_bits(),
        flags,
    ] {
        kernarg.push_u32(value);
    }
    command_buffer.dispatch(
        &function,
        [n.div_ceil(ROWS_PER_GROUP * tiles), m.div_ceil(ROWS_PER_GROUP), 1],
        [32 * WAVES, 1, 1],
        kernarg,
        "QmvWmma",
    );
    Ok(())
}
