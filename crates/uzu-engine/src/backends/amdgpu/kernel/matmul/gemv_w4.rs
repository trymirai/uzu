//! Quantized matmul for one activation row (decode) in the checkpoint layout of uzu's quantized models: the
//! native kernel `kernel/native/gemv_w4.clcpp`. Against the MSL GEMV it loads a lane's scales and zero points
//! for all its rows with one vector load each and prefetches the next step (codes, activations, metadata)
//! before the math of the current one; the MSL GEMV waits on scalar metadata loads per group, which made zero
//! points cost 17-35% of the decode matmul time. The output transform (scale, RHT, bias) runs in the epilogue
//! as in the MSL GEMV. `UZU_AMDGPU_GEMV_NATIVE=0` keeps the MSL GEMV; `UZU_AMDGPU_GEMV_NATIVE_ROWS=4|8`
//! overrides the rows per wave.

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
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulShape, QuantParamsLayout},
        },
    },
    data_type::DataType,
};

/// Waves of a workgroup (`GEMV_WAVES` in the kernel).
const WAVES: u32 = 8;
/// A lane reads 32 codes of a row per step.
const K_MULTIPLE: u32 = 32;
const RHT_BLOCK: u32 = 32;

const FLAG_SIGNED_CODES: u32 = 1;
const FLAG_SCALE: u32 = 2;
const FLAG_BIAS: u32 = 4;
const FLAG_RHT: u32 = 8;

/// 0 / 1, or UNSET until the first call reads `UZU_AMDGPU_GEMV_NATIVE`.
static ENABLED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(UNSET);
const UNSET: u8 = 2;

pub fn enabled() -> bool {
    use std::sync::atomic::Ordering;
    match ENABLED.load(Ordering::Relaxed) {
        UNSET => {
            let enabled = std::env::var("UZU_AMDGPU_GEMV_NATIVE").map_or(true, |value| value != "0");
            ENABLED.store(enabled as u8, Ordering::Relaxed);
            enabled
        },
        value => value == 1,
    }
}

/// Switches the route between encodes, for in-process A/B probes (`perf_token_matmuls_qwen9b`).
#[cfg(test)]
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled as u8, std::sync::atomic::Ordering::Relaxed);
}

/// Output rows per wave: 4 (32 per workgroup), 8 for very tall matrices. With the cooperative metadata load
/// (interleaved A/B on the 890M, 9B token shapes) 4 rows are 3-8% faster on the 4096-24576-row projections and
/// 8 rows 7% faster on the 248k-row readout.
fn rows(n: u32) -> u32 {
    static OVERRIDE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    let rows = *OVERRIDE.get_or_init(|| {
        std::env::var("UZU_AMDGPU_GEMV_NATIVE_ROWS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|rows| matches!(rows, 4 | 8))
    });
    rows.unwrap_or(if n >= 65536 {
        8
    } else {
        4
    })
}

/// Scales and zero points loaded once per workgroup and step and handed out through LDS (`_meta` entries);
/// `UZU_AMDGPU_GEMV_NATIVE_META=0` keeps the per-lane loads.
fn cooperative_metadata() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("UZU_AMDGPU_GEMV_NATIVE_META").map_or(true, |value| value != "0"))
}

fn address(buffer: impl BufferRef<Backend = Amdgpu>) -> u64 {
    let (buffer, range) = buffer.parts();
    buffer.device_address() + range.start as u64
}

pub fn supports(
    shape: &MatmulShape,
    data_types: [DataType; 3],
) -> bool {
    let rows = rows(shape.n);
    !native::GEMV_W4.is_empty()
        && enabled()
        && shape.m == 1
        && matches!(
            shape.b_prologue,
            GemmBPrologueKind::ScaleZeroPointDequant | GemmBPrologueKind::ScaleSymmetricDequant
        )
        && shape.b_bits == Some(4)
        && matches!(shape.b_group_size, Some(32 | 64))
        && shape.params_layout == Some(QuantParamsLayout::GroupOutput)
        && shape.b_transpose
        && shape.b_leading_dimension.is_none()
        && shape.a_full_precision
        && !shape.gathered
        && data_types.iter().all(|&data_type| data_type == DataType::BF16)
        && shape.k.is_multiple_of(K_MULTIPLE)
        && shape.b_group_size.is_some_and(|group| shape.k.is_multiple_of(group))
        && shape.n >= rows
        && shape.n.is_multiple_of(rows)
        && !shape.d_transform.intersects(GemmDTransform::ACCUMULATE | GemmDTransform::SOFT_CAP)
        && (!shape.d_transform.contains(GemmDTransform::RHT) || shape.n.is_multiple_of(RHT_BLOCK))
}

/// Codes and activations are read as 16-byte vectors, a lane's scales as one `2 * rows`-byte vector and its zero
/// points as one `rows / 2`-byte word: group-major metadata with the group stride a multiple of the rows.
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
    let rows = rows(arguments.n) as u64;
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
    let scale_strides = quantized.params.scale_strides();
    let zero_points_aligned = match quantized.correction.zero_points() {
        Some(&zero_points) => {
            let strides = quantized.zero_point_strides();
            strides.output_stride == 1
                && (strides.group_stride as u64).is_multiple_of(rows)
                && address(zero_points).is_multiple_of(rows / 2)
        },
        None => quantized.correction.biases().is_none(),
    };
    (address(*values) + *offset as u64).is_multiple_of(16)
        && address(quantized.codes).is_multiple_of(16)
        && scale_strides.output_stride == 1
        && (scale_strides.group_stride as u64).is_multiple_of(rows)
        && address(quantized.scales).is_multiple_of(2 * rows)
        && zero_points_aligned
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
            path: "GemvW4",
            reason: "int8 activations",
        });
    };
    let MatmulB::Quantized(quantized) = b else {
        return Err(MatmulError::IncompatibleA {
            path: "GemvW4",
            reason: "full-precision weights",
        });
    };
    let rows = rows(n);
    let prologue = if quantized.correction.zero_points().is_some() {
        "zero_point"
    } else {
        "symmetric"
    };
    let entry = format!(
        "gemv_w4_g{}_{prologue}_r{rows}{}",
        quantized.group_size,
        if cooperative_metadata() {
            "_meta"
        } else {
            ""
        }
    );
    let function = command_buffer.context().function(&[native::GEMV_W4], &entry).map_err(MatmulError::BackendError)?;

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
    command_buffer.dispatch(&function, [n.div_ceil(WAVES * rows), m, 1], [32 * WAVES, 1, 1], kernarg, "GemvW4");
    Ok(())
}
