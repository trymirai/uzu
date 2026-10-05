//! Quantized GEMM for prefill (M >= 48) on WMMA: the native kernel `kernel/native/gemm_w4.clcpp`, 128 x 128
//! output tiles of 8 waves with A and dequantized weights staged in LDS (double buffered). The MSL GEMM builds
//! for AMD only with 32x32 and 64x64 tiles of 4 SIMD groups and reaches ~2 TFLOPS on 9B prefill; this kernel
//! 3-6 TFLOPS. The output transform (scale, RHT, bias) runs in the epilogue as in the MSL GEMV.
//! `UZU_AMDGPU_GEMM_NATIVE=0` keeps the MSL GEMM.

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
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulShape},
        },
    },
    data_type::DataType,
};

/// Waves of a workgroup (`GEMM_WAVES` in the kernel) and its output tile.
const WAVES: u32 = 8;
const TILE: u32 = 128;
/// K advances 32 per step; a wave's 32 output columns are one Hadamard block.
const K_MULTIPLE: u32 = 32;
const N_MULTIPLE: u32 = 32;

const FLAG_SIGNED_CODES: u32 = 1;
const FLAG_SCALE: u32 = 2;
const FLAG_BIAS: u32 = 4;
const FLAG_RHT: u32 = 8;

/// 0 / 1, or UNSET until the first call reads `UZU_AMDGPU_GEMM_NATIVE`.
static ENABLED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(UNSET);
const UNSET: u8 = 2;

pub fn enabled() -> bool {
    use std::sync::atomic::Ordering;
    match ENABLED.load(Ordering::Relaxed) {
        UNSET => {
            let enabled = std::env::var("UZU_AMDGPU_GEMM_NATIVE").map_or(true, |value| value != "0");
            ENABLED.store(enabled as u8, Ordering::Relaxed);
            enabled
        },
        value => value == 1,
    }
}

/// Switches the route between encodes, for in-process A/B probes.
#[cfg(test)]
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled as u8, std::sync::atomic::Ordering::Relaxed);
}

fn address(buffer: impl BufferRef<Backend = Amdgpu>) -> u64 {
    let (buffer, range) = buffer.parts();
    buffer.device_address() + range.start as u64
}

pub fn supports(
    shape: &MatmulShape,
    data_types: [DataType; 3],
) -> bool {
    !native::GEMM_W4.is_empty()
        && enabled()
        && matches!(
            shape.b_prologue,
            GemmBPrologueKind::ScaleZeroPointDequant | GemmBPrologueKind::ScaleSymmetricDequant
        )
        && shape.b_bits == Some(4)
        && matches!(shape.b_group_size, Some(32 | 64))
        && shape.b_transpose
        && shape.b_leading_dimension.is_none()
        && shape.a_full_precision
        && !shape.gathered
        && data_types.iter().all(|&data_type| data_type == DataType::BF16)
        && shape.k.is_multiple_of(K_MULTIPLE)
        && shape.b_group_size.is_some_and(|group| shape.k.is_multiple_of(group))
        && shape.n.is_multiple_of(N_MULTIPLE)
        && !shape.d_transform.intersects(GemmDTransform::ACCUMULATE | GemmDTransform::SOFT_CAP)
}

/// A rows are read as 16-byte vectors and code rows as 8-byte words.
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
    let MatmulA::FullPrecision {
        values,
        offset,
    } = &arguments.a
    else {
        return false;
    };
    let MatmulB::Quantized(quantized) = &arguments.b else {
        return false;
    };
    (address(*values) + *offset as u64).is_multiple_of(16)
        && address(quantized.codes).is_multiple_of(16)
        && (quantized.correction.zero_points().is_some() || quantized.correction.biases().is_none())
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
    let MatmulA::FullPrecision {
        values,
        offset,
    } = a
    else {
        return Err(MatmulError::IncompatibleA {
            path: "GemmW4",
            reason: "int8 activations",
        });
    };
    let MatmulB::Quantized(quantized) = b else {
        return Err(MatmulError::IncompatibleA {
            path: "GemmW4",
            reason: "full-precision weights",
        });
    };
    let prologue = if quantized.correction.zero_points().is_some() {
        "zero_point"
    } else {
        "symmetric"
    };
    let entry = format!("gemm_w4_g{}_{prologue}", quantized.group_size);
    let function = command_buffer.context().function(&[native::GEMM_W4], &entry).map_err(MatmulError::BackendError)?;

    let rht_factors = d_transform.rht_factors;
    let bias = d_transform.bias;
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

    // kernel arguments as qmv_wmma (biases and activation scales unused)
    let mut kernarg = Kernarg::new();
    kernarg.push_address(address(quantized.codes));
    kernarg.push_address(address(quantized.scales));
    kernarg.push_address(quantized.correction.zero_points().map_or(0, |&buffer| address(buffer)));
    kernarg.push_address(0);
    kernarg.push_address(address(values) + offset as u64);
    kernarg.push_address(0);
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
    command_buffer.dispatch(&function, [n.div_ceil(TILE), m.div_ceil(TILE), 1], [32 * WAVES, 1, 1], kernarg, "GemmW4");
    Ok(())
}
