#![cfg(backend = "metal")]

//! QTIP race profile: per-leaf-family GPU timing of the physical Q8-table QTIP
//! kernels against candidate kernels, with bit-exact output checks against the
//! accepted physical kernels and an adjacent Uzu G64 (ScaleBias, 4-bit, g64)
//! GEMM reference. Uniform pseudo-random code bytes model the ~16-bit state
//! entropy of the real package.
//!
//! Each timing encodes the kernel `QTIP_RACE_INNER` times in one command buffer
//! (amortising per-command-buffer overhead like the real model), takes the
//! median of `QTIP_RACE_REPS` command buffers, and divides by the inner count.
//!
//! Batch 8 and 16 cases run on 32-padded activations; the baseline for those is
//! the production path (B16 kernel on the first 16 rows, row-major output, then
//! the batch-rows transpose kernel). Variants named `diag_*` are speed-only
//! probes and are excluded from the bit-exact check.

use half::{bf16, f16};
use proc_macros::uzu_test;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{
    backends::{
        common::{
            Allocation, Backend, Encoder,
            gpu_types::QuantizationMethod,
            kernel::{Kernels, matmul::MatmulKernel},
        },
        metal::*,
    },
    data_type::DataType,
    tests::{
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
        matmul::{QuantBuffers, QuantInput, quant_arguments},
    },
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Geometry {
    V4,
    V2K2,
    V2K3,
    V4K3,
    V8,
}

#[derive(Clone, Copy)]
struct Family {
    name: &'static str,
    geometry: Geometry,
    rows: u32,
    columns: u32,
    leaves: u32,
}

/// Physical Qwen3.8 S package families (audited from safetensor spec metadata
/// and code shapes on 2026-09-01): 166 V4-k2, 9 V2-k2, 97 V2-k3.
const FAMILIES: [Family; 12] = [
    Family { name: "in_proj", geometry: Geometry::V4, rows: 16480, columns: 5120, leaves: 4 },
    Family { name: "in_proj", geometry: Geometry::V2K3, rows: 16480, columns: 5120, leaves: 44 },
    Family { name: "gate", geometry: Geometry::V4, rows: 6144, columns: 5120, leaves: 3 },
    Family { name: "gate", geometry: Geometry::V2K3, rows: 6144, columns: 5120, leaves: 13 },
    Family { name: "qkv", geometry: Geometry::V2K3, rows: 8192, columns: 5120, leaves: 16 },
    Family { name: "mlp_down", geometry: Geometry::V2K2, rows: 5120, columns: 17408, leaves: 3 },
    Family { name: "mlp_down", geometry: Geometry::V4, rows: 5120, columns: 17408, leaves: 61 },
    Family { name: "mlp_up", geometry: Geometry::V2K2, rows: 34816, columns: 5120, leaves: 3 },
    Family { name: "mlp_up", geometry: Geometry::V4, rows: 34816, columns: 5120, leaves: 37 },
    Family { name: "mlp_up", geometry: Geometry::V2K3, rows: 34816, columns: 5120, leaves: 24 },
    Family { name: "out_proj", geometry: Geometry::V2K2, rows: 5120, columns: 6144, leaves: 3 },
    Family { name: "out_proj", geometry: Geometry::V4, rows: 5120, columns: 6144, leaves: 61 },
];

struct Case {
    codes: Allocation<Metal>,
    codebook: Allocation<Metal>,
    /// V4 only: same table re-laid as [components 0,1 for all states][components 2,3 for all states]
    codebook_split: Allocation<Metal>,
    /// V4 only: K-packed activation halves (columns 4g,4g+1 | 4g+2,4g+3), [padded_batch x columns/2]
    activations_lo: Allocation<Metal>,
    activations_hi: Allocation<Metal>,
    codebook_scale: f32,
    activations: Allocation<Metal>,
    activation_scales: Allocation<Metal>,
    scales: Allocation<Metal>,
    gains: Allocation<Metal>,
    rows: u32,
    groups: u32,
    bytes_per_row: u32,
    /// tokens actually computed / written
    active_batch: u32,
    host: HostData,
}

/// Host copies for the CPU oracle.
struct HostData {
    codes: Vec<u8>,
    codebook: Vec<i8>,
    activations: Vec<i8>,
    activation_scales: Vec<f32>,
    scales: Vec<f16>,
    gains: Vec<bf16>,
    columns: u32,
    geometry: Geometry,
}

/// V2 table lookup honouring QTIP_RACE_TABLE=v2sign14 (16K base rows; bit 15 negates both components, bit 14 component 0)
fn v2_table_value(host: &HostData, state: usize, component: usize) -> i8 {
    let state = match std::env::var("QTIP_RACE_STATE_BITS").ok().as_deref() {
        Some("15") => state & 0x7FFF,
        Some("14") => state & 0x3FFF,
        _ => state,
    };
    if std::env::var("QTIP_RACE_TABLE").ok().as_deref() == Some("v2sign14") {
        let raw = host.codebook[(state & 0x3FFF) * 2 + component];
        let flipped = if state & 0x8000 != 0 { -raw } else { raw };
        if state & 0x4000 != 0 && component == 0 { -flipped } else { flipped }
    } else {
        host.codebook[state * 2 + component]
    }
}

/// Exact CPU reference for rows [0, row_limit) and tokens [0, active_batch):
/// int32 dot products of the decoded INT8 weights with the signed-A8
/// activations, then the kernels' fp32 epilogue and bf16 rounding.
#[derive(Clone, Copy, PartialEq)]
enum TableMode {
    Plain,
    Anti15,
    Sign12,
    Sign14,
}

/// QTIP_RACE_TABLE=anti15 (15-bit index, bit 15 negates the row) or sign12 (12-bit index, bits 12..15 flip
/// components 0..3) models the structured-table representations; the oracle mirrors the kernels' decode.
fn table_mode() -> TableMode {
    match std::env::var("QTIP_RACE_TABLE").ok().as_deref() {
        Some("anti15") => TableMode::Anti15,
        Some("sign12") => TableMode::Sign12,
        Some("sign14") => TableMode::Sign14,
        _ => TableMode::Plain,
    }
}

/// QTIP_RACE_STATE_BITS=15 models the L=15 refit: V4 states are masked to 15 bits
fn state_mask() -> usize {
    match table_mode() {
        TableMode::Anti15 => 0x7FFF,
        TableMode::Sign12 => 0x0FFF,
        TableMode::Sign14 => 0x3FFF,
        TableMode::Plain => match std::env::var("QTIP_RACE_STATE_BITS").ok().as_deref() {
            Some("15") => 0x7FFF,
            _ => 0xFFFF,
        },
    }
}

fn cpu_reference(
    case: &Case,
    row_limit: usize,
) -> Vec<bf16> {
    let host = &case.host;
    let columns = host.columns as usize;
    let rows = case.rows as usize;
    let batch = case.active_batch as usize;
    let bytes_per_row = case.bytes_per_row as usize;
    let mut weights = vec![0i32; columns];
    let mut output = vec![bf16::ZERO; batch * rows];
    for row in 0..row_limit.min(rows) {
        let row_codes = &host.codes[row * bytes_per_row..(row + 1) * bytes_per_row];
        for column in 0..columns {
            let value = match host.geometry {
                Geometry::V4 => {
                    let block = column / 64;
                    let gib = (column % 64) / 4;
                    let seq = &row_codes[block * 17..block * 17 + 17];
                    let full = match gib {
                        0 => (seq[0] as usize) | ((seq[1] as usize) << 8),
                        1 => ((seq[0] as usize) << 8) | (seq[2] as usize),
                        _ => ((seq[gib] as usize) << 8) | (seq[gib + 1] as usize),
                    };
                    let state = full & state_mask();
                    let raw = host.codebook[state * 4 + column % 4];
                    match table_mode() {
                        TableMode::Sign14 => {
                            let flipped = if full & 0x8000 != 0 { -raw } else { raw };
                            if full & 0x4000 != 0 && column % 4 < 2 { -flipped } else { flipped }
                        },
                        TableMode::Anti15 => if full & 0x8000 != 0 { -raw } else { raw },
                        TableMode::Sign12 => if (full >> (12 + column % 4)) & 1 != 0 { -raw } else { raw },
                        TableMode::Plain => raw,
                    }
                },
                Geometry::V8 => {
                    let group = column / 8;
                    let start = (group / 8) * 17 + (group % 8) * 2;
                    let at = |offset: usize| row_codes.get(start + offset).copied().unwrap_or(0) as u32;
                    let window = at(0) << 16 | at(1) << 8 | at(2);
                    let state = ((window >> 4) & 0xFFFFF) as usize;
                    let base = state & 0xFFF;
                    let c = column % 8;
                    let raw = host.codebook[base * 8 + c];
                    if (state >> (12 + c)) & 1 != 0 { -raw } else { raw }
                },
                Geometry::V4K3 => {
                    let group = column / 4;
                    let block = group / 16;
                    let bit = (group % 16) * 12;
                    let start = block * 25 + bit / 8;
                    // the 20-bit window never crosses the 25-byte block; the fourth byte only feeds shifted-out bits
                    let at = |offset: usize| row_codes.get(start + offset).copied().unwrap_or(0) as u32;
                    let window = at(0) << 24 | at(1) << 16 | at(2) << 8 | at(3);
                    let state = ((window >> (12 - (bit % 8) as u32)) & 0xFFFFF) as usize;
                    let (mask, sign17) = match std::env::var("QTIP_RACE_V4K3").ok().as_deref() {
                        Some("sign17") => (0x1FFFF, true),
                        _ => (0xFFFFF, false),
                    };
                    let state = state & mask;
                    let base = state & 0x7FFF;
                    let h = state >> 15;
                    let c = column % 4;
                    if sign17 {
                        let raw = host.codebook[base * 4 + c];
                        let v = if h & 1 != 0 { -raw } else { raw };
                        if h & 2 != 0 && c < 2 { -v } else { v }
                    } else {
                        // negations on bits 15..18 apply to the component before the swap; bit 19 swaps 0 <-> 1
                        let source = if h & 16 != 0 && c < 2 { 1 - c } else { c };
                        let raw = host.codebook[base * 4 + source];
                        if (h >> source) & 1 != 0 { -raw } else { raw }
                    }
                },
                Geometry::V2K2 => {
                    let byte = column / 4;
                    let b0 = row_codes[byte] as usize;
                    let b1 = row_codes[byte + 1] as usize;
                    let b2 = row_codes[byte + 2] as usize;
                    let state = if column % 4 < 2 {
                        (b0 << 8) | b1
                    } else {
                        ((b0 << 12) | (b1 << 4) | (b2 >> 4)) & 0xFFFF
                    };
                    v2_table_value(host, state, column % 2)
                },
                Geometry::V2K3 => {
                    let group = column / 2;
                    let bit = group * 6;
                    let byte = bit >> 3;
                    let shift = bit & 7;
                    // the fourth byte never contributes for shift <= 7; guard the row end
                    let at = |offset: usize| row_codes.get(byte + offset).copied().unwrap_or(0) as u32;
                    let window = (at(0) << 24) | (at(1) << 16) | (at(2) << 8) | at(3);
                    let state = ((window >> (16 - shift)) & 0xFFFF) as usize;
                    v2_table_value(host, state, column % 2)
                },
            };
            weights[column] = value as i32;
        }
        let weight_scale = (host.scales[row].to_f32() * host.gains[row].to_f32()) * case.codebook_scale;
        for token in 0..batch {
            let activations = &host.activations[token * columns..(token + 1) * columns];
            let sum: i32 = weights.iter().zip(activations).map(|(w, a)| w * (*a as i32)).sum();
            output[token * rows + row] = bf16::from_f32((sum as f32 * weight_scale) * host.activation_scales[token]);
        }
    }
    output
}

fn build_case(
    context: &MetalContext,
    family_in: &Family,
    active_batch: u32,
    padded_batch: u32,
    seed: u64,
) -> Case {
    // QTIP_RACE_V4K3=sym20|sign17 models the k=3 leaves as V4 k=3 (12-bit transitions, 25-byte blocks)
    let mut family_v4k3 = *family_in;
    if std::env::var("QTIP_RACE_V4K3").is_ok() && matches!(family_in.geometry, Geometry::V2K3) {
        family_v4k3.geometry = Geometry::V4K3;
    }
    if std::env::var("QTIP_RACE_V8").is_ok() && matches!(family_in.geometry, Geometry::V4) {
        family_v4k3.geometry = Geometry::V8;
    }
    let family = &family_v4k3;
    let mut rng = SmallRng::seed_from_u64(seed);
    let (groups, bytes_per_row, vector_width) = match family.geometry {
        Geometry::V4 => (family.columns / 4, family.columns / 64 * 17, 4u32),
        Geometry::V4K3 => (family.columns / 4, family.columns / 64 * 25, 4u32),
        Geometry::V8 => (family.columns / 8, family.columns / 64 * 17, 8u32),
        Geometry::V2K2 => {
            let groups = family.columns / 2;
            (groups, (16 + (groups - 1) * 4).div_ceil(8), 2)
        },
        Geometry::V2K3 => {
            let groups = family.columns / 2;
            (groups, (16 + (groups - 1) * 6).div_ceil(8), 2)
        },
    };
    let codes: Vec<u8> = (0..(family.rows as usize * bytes_per_row as usize)).map(|_| rng.random::<u8>()).collect();
    let codebook: Vec<i8> =
        (0..(65_536 * vector_width as usize)).map(|_| rng.random_range(-127i16..=127) as i8).collect();
    let mut codebook_split = vec![0i8; codebook.len()];
    if vector_width == 4 {
        for state in 0..65_536usize {
            codebook_split[2 * state] = codebook[4 * state];
            codebook_split[2 * state + 1] = codebook[4 * state + 1];
            codebook_split[131_072 + 2 * state] = codebook[4 * state + 2];
            codebook_split[131_072 + 2 * state + 1] = codebook[4 * state + 3];
        }
    }
    let mut activations: Vec<i8> = vec![0; padded_batch as usize * family.columns as usize];
    for value in activations.iter_mut().take(active_batch as usize * family.columns as usize) {
        *value = rng.random_range(-127i16..=127) as i8;
    }
    let half = family.columns as usize / 2;
    let mut activations_lo = vec![0i8; padded_batch as usize * half];
    let mut activations_hi = vec![0i8; padded_batch as usize * half];
    for token in 0..padded_batch as usize {
        for column in 0..family.columns as usize {
            let value = activations[token * family.columns as usize + column];
            let index = token * half + (column / 4) * 2 + (column & 1);
            if column & 2 == 0 {
                activations_lo[index] = value;
            } else {
                activations_hi[index] = value;
            }
        }
    }
    let mut activation_scales: Vec<f32> = vec![1.0; padded_batch as usize];
    for value in activation_scales.iter_mut().take(active_batch as usize) {
        *value = rng.random_range(0.002f32..0.02);
    }
    let scales: Vec<f16> = (0..family.rows as usize).map(|_| f16::from_f32(rng.random_range(0.01f32..0.1))).collect();
    let gains: Vec<bf16> = (0..family.rows as usize).map(|_| bf16::from_f32(rng.random_range(0.5f32..1.5))).collect();
    Case {
        codes: alloc_allocation_with_data::<Metal, u8>(context, &codes),
        codebook: alloc_allocation_with_data::<Metal, i8>(context, &codebook),
        codebook_split: alloc_allocation_with_data::<Metal, i8>(context, &codebook_split),
        activations_lo: alloc_allocation_with_data::<Metal, i8>(context, &activations_lo),
        activations_hi: alloc_allocation_with_data::<Metal, i8>(context, &activations_hi),
        codebook_scale: 4.0 / 127.0,
        activations: alloc_allocation_with_data::<Metal, i8>(context, &activations),
        activation_scales: alloc_allocation_with_data::<Metal, f32>(context, &activation_scales),
        scales: alloc_allocation_with_data::<Metal, f16>(context, &scales),
        gains: alloc_allocation_with_data::<Metal, bf16>(context, &gains),
        rows: family.rows,
        groups,
        bytes_per_row,
        active_batch,
        host: HostData {
            codes,
            codebook,
            activations,
            activation_scales,
            scales,
            gains,
            columns: family.columns,
            geometry: family.geometry,
        },
    }
}

/// (case, output [active_batch x rows], scratch [rows x 16 bf16 | rows x batch int32 partials], encoder)
type Runner<'a> = Box<dyn Fn(&Case, &mut Allocation<Metal>, &mut Allocation<Metal>, &mut Encoder<Metal>) + 'a>;

/// physical B32 / B64 batch-rows kernels (no active_batch argument)
macro_rules! runner_base {
    ($kernel:expr) => {{
        let kernel = &$kernel;
        Box::new(
            move |case: &Case, output: &mut Allocation<Metal>, _scratch: &mut Allocation<Metal>, encoder: &mut Encoder<Metal>| {
                kernel.encode(
                    &case.codes,
                    &case.codebook,
                    &case.activations,
                    &case.activation_scales,
                    &case.scales,
                    &case.gains,
                    output,
                    case.codebook_scale,
                    case.rows,
                    case.groups,
                    case.bytes_per_row,
                    encoder,
                )
            },
        ) as Runner<'_>
    }};
}

/// race kernels (active_batch argument, partials scratch)
macro_rules! runner {
    ($kernel:expr) => {{
        let kernel = &$kernel;
        Box::new(
            move |case: &Case, output: &mut Allocation<Metal>, scratch: &mut Allocation<Metal>, encoder: &mut Encoder<Metal>| {
                kernel.encode(
                    &case.codes,
                    &case.codebook,
                    &case.activations,
                    &case.activation_scales,
                    &case.scales,
                    &case.gains,
                    output,
                    &mut *scratch,
                    case.codebook_scale,
                    case.rows,
                    case.groups,
                    case.bytes_per_row,
                    case.active_batch,
                    encoder,
                )
            },
        ) as Runner<'_>
    }};
}

/// component-split dual dispatch: pass 0 on the (0,1) half, pass 1 on the (2,3) half
macro_rules! runner_cs {
    ($pass0:expr, $pass1:expr) => {{
        let pass0 = &$pass0;
        let pass1 = &$pass1;
        Box::new(
            move |case: &Case, output: &mut Allocation<Metal>, scratch: &mut Allocation<Metal>, encoder: &mut Encoder<Metal>| {
                pass0.encode(
                    &case.codes,
                    &case.codebook_split,
                    &case.activations_lo,
                    &case.activation_scales,
                    &case.scales,
                    &case.gains,
                    &mut *output,
                    &mut *scratch,
                    case.codebook_scale,
                    case.rows,
                    case.groups,
                    case.bytes_per_row,
                    case.active_batch,
                    encoder,
                );
                pass1.encode(
                    &case.codes,
                    &case.codebook_split,
                    &case.activations_hi,
                    &case.activation_scales,
                    &case.scales,
                    &case.gains,
                    output,
                    &mut *scratch,
                    case.codebook_scale,
                    case.rows,
                    case.groups,
                    case.bytes_per_row,
                    case.active_batch,
                    encoder,
                );
            },
        ) as Runner<'_>
    }};
}

/// production small-suffix baseline: B16 kernel (row-major [rows x 16]) + transpose to [active_batch x rows]
macro_rules! runner_base16 {
    ($kernel:expr, $transpose:expr) => {{
        let kernel = &$kernel;
        let transpose = &$transpose;
        Box::new(
            move |case: &Case, output: &mut Allocation<Metal>, scratch: &mut Allocation<Metal>, encoder: &mut Encoder<Metal>| {
                kernel.encode(
                    &case.codes,
                    &case.codebook,
                    &case.activations,
                    &case.activation_scales,
                    &case.scales,
                    &case.gains,
                    &mut *scratch,
                    case.codebook_scale,
                    case.rows,
                    case.groups,
                    case.bytes_per_row,
                    encoder,
                );
                transpose.encode(&*scratch, output, case.active_batch, 16, case.rows, encoder);
            },
        ) as Runner<'_>
    }};
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    if std::env::var("QTIP_RACE_STAT").map_or(false, |value| value == "min") {
        values[0]
    } else {
        values[values.len() / 2]
    }
}

#[uzu_test]
fn qtip_race_profile() {
    let context = crate::tests::util::shared_metal_context();
    assert!(context.supports_mxu(), "race profile requires MXU support");
    let batches: Vec<u32> = std::env::var("QTIP_RACE_BATCHES")
        .map(|value| value.split(',').map(|item| item.trim().parse::<u32>().expect("batch")).collect())
        .unwrap_or_else(|_| vec![32, 64]);
    let reps = std::env::var("QTIP_RACE_REPS").map_or(5usize, |value| value.parse().expect("reps"));
    let inner = std::env::var("QTIP_RACE_INNER").map_or(4usize, |value| value.parse().expect("inner"));
    let family_filter = std::env::var("QTIP_RACE_FAMILY").ok();
    let variant_filter = std::env::var("QTIP_RACE_VARIANTS").ok();
    let skip_g64 = std::env::var("QTIP_RACE_SKIP_G64").is_ok();
    let skip_diag = std::env::var("QTIP_RACE_SKIP_DIAG").is_ok();
    let oracle_rows = std::env::var("QTIP_RACE_ORACLE").ok().map(|value| value.parse::<usize>().expect("oracle rows"));
    let rows_override: Vec<u32> = std::env::var("QTIP_RACE_ROWS")
        .map(|value| value.split(',').map(|item| item.trim().parse::<u32>().expect("rows")).collect())
        .unwrap_or_default();

    macro_rules! kernel {
        ($name:ident) => {
            $name::new(&context).expect(stringify!($name))
        };
    }
    let transpose = kernel!(QtipRowsBatchToBatchRowsBf16MetalKernel);
    let base_v4_b16 = kernel!(QtipGaussianPhysicalQ8V4A8DirectB16MetalKernel);
    let base_v4_b32 = kernel!(QtipGaussianPhysicalQ8V4A8DirectB32BatchRowsMetalKernel);
    let base_v4_b64 = kernel!(QtipGaussianPhysicalQ8V4A8DirectB64BatchRowsMetalKernel);
    let base_k2_b16 = kernel!(QtipGaussianPhysicalQ8V2A8DirectK2B16MetalKernel);
    let base_k2_b32 = kernel!(QtipGaussianPhysicalQ8V2A8DirectK2B32BatchRowsMetalKernel);
    let base_k2_b64 = kernel!(QtipGaussianPhysicalQ8V2A8DirectK2B64BatchRowsMetalKernel);
    let base_k3_b16 = kernel!(QtipGaussianPhysicalQ8V2A8DirectK3B16MetalKernel);
    let base_k3_b32 = kernel!(QtipGaussianPhysicalQ8V2A8DirectK3B32BatchRowsMetalKernel);
    let base_k3_b64 = kernel!(QtipGaussianPhysicalQ8V2A8DirectK3B64BatchRowsMetalKernel);

    let v4_pf2_sg4_b32 = kernel!(QtipRaceV4Pf2Sg4B32MetalKernel);
    let v4_pf2_sg2_b32 = kernel!(QtipRaceV4Pf2Sg2B32MetalKernel);
    let v4_pf2_sg4_b64 = kernel!(QtipRaceV4Pf2Sg4B64MetalKernel);
    let v4_pf2_sg2_b64 = kernel!(QtipRaceV4Pf2Sg2B64MetalKernel);
    let v4_pf2_sg8_b64 = kernel!(QtipRaceV4Pf2Sg8B64MetalKernel);
    let l15_pf2_sg4_b32 = kernel!(QtipRaceV4L15Pf2Sg4B32MetalKernel);
    let l15_pf2_sg2_b32 = kernel!(QtipRaceV4L15Pf2Sg2B32MetalKernel);
    let l15_r2_pf2_sg2_b32 = kernel!(QtipRaceV4L15R2Pf2Sg2B32MetalKernel);
    let l15_pf2_sg2_b64 = kernel!(QtipRaceV4L15Pf2Sg2B64MetalKernel);
    let l15_pf2_sg4_b64 = kernel!(QtipRaceV4L15Pf2Sg4B64MetalKernel);
    let l15_pf2_sg8_b64 = kernel!(QtipRaceV4L15Pf2Sg8B64MetalKernel);
    let l15_t2_pf0_b16 = kernel!(QtipRaceV4L15T2Pf0Sg2B16MetalKernel);
    let l15_t2_pf2_b16 = kernel!(QtipRaceV4L15T2Pf2Sg2B16MetalKernel);
    let l15_t4_pf1_b16 = kernel!(QtipRaceV4L15T4Pf1Sg2B16MetalKernel);
    let v4_sign12_t2_b16 = kernel!(QtipRaceV4Sign12T2Pf2Sg2B16MetalKernel);
    let k3_l15_sg2_b32 = kernel!(QtipRaceK3L15Pf2Sg2B32MetalKernel);
    let k3_l15_pf0sg4_b32 = kernel!(QtipRaceK3L15Pf0Sg4B32MetalKernel);
    let k3_l15_r2_b32 = kernel!(QtipRaceK3L15R2Pf2Sg2B32MetalKernel);
    let k3_l15_sg4_b64 = kernel!(QtipRaceK3L15Pf2Sg4B64MetalKernel);
    let k3_l15_sg2_b64 = kernel!(QtipRaceK3L15Pf2Sg2B64MetalKernel);
    let k3_l15_t2_b16 = kernel!(QtipRaceK3L15T2Pf0Sg2B16MetalKernel);
    let k3_l15_t4_b16 = kernel!(QtipRaceK3L15T4Pf0Sg2B16MetalKernel);
    let k2_l15_sg2_b32 = kernel!(QtipRaceK2L15Pf2Sg2B32MetalKernel);
    let k2_l15_pf0sg4_b32 = kernel!(QtipRaceK2L15Pf0Sg4B32MetalKernel);
    let k2_l15_r2_b32 = kernel!(QtipRaceK2L15R2Pf2Sg2B32MetalKernel);
    let k2_l15_sg2_b64 = kernel!(QtipRaceK2L15Pf2Sg2B64MetalKernel);
    let k2_l15_t2_b16 = kernel!(QtipRaceK2L15T2Pf0Sg2B16MetalKernel);
    let k2_l15_t2pf2_b16 = kernel!(QtipRaceK2L15T2Pf2Sg2B16MetalKernel);
    let v4_s14_as_sg16_b64 = kernel!(QtipRaceV4Sign14AsPf1Sg16B64MetalKernel);
    let v4_s14_t2_b16 = kernel!(QtipRaceV4Sign14T2Pf2Sg2B16MetalKernel);
    let v4_anti_as_sg16_b64 = kernel!(QtipRaceV4AntiAsPf1Sg16B64MetalKernel);
    let v4_anti_t2_b16 = kernel!(QtipRaceV4AntiT2Pf2Sg2B16MetalKernel);
    let v4_sign14_sg2_b32 = kernel!(QtipRaceV4Sign14Pf2Sg2B32MetalKernel);
    let v4_sign14_sg4_b32 = kernel!(QtipRaceV4Sign14Pf2Sg4B32MetalKernel);
    let v4_sign14_sg2_b64 = kernel!(QtipRaceV4Sign14Pf2Sg2B64MetalKernel);
    let v4_sign14_sg4_b64 = kernel!(QtipRaceV4Sign14Pf2Sg4B64MetalKernel);
    let v4_anti_sg2_b32 = kernel!(QtipRaceV4AntiPf2Sg2B32MetalKernel);
    let v4_anti_sg4_b32 = kernel!(QtipRaceV4AntiPf2Sg4B32MetalKernel);
    let v4_anti_sg2_b64 = kernel!(QtipRaceV4AntiPf2Sg2B64MetalKernel);
    let v4_anti_sg4_b64 = kernel!(QtipRaceV4AntiPf2Sg4B64MetalKernel);
    let v4_sign12_sg2_b32 = kernel!(QtipRaceV4Sign12Pf2Sg2B32MetalKernel);
    let v4_sign12_sg4_b32 = kernel!(QtipRaceV4Sign12Pf2Sg4B32MetalKernel);
    let v4_sign12_sg2_b64 = kernel!(QtipRaceV4Sign12Pf2Sg2B64MetalKernel);
    let v4_sign12_sg4_b64 = kernel!(QtipRaceV4Sign12Pf2Sg4B64MetalKernel);
    let v4_l15as_sg8_b32 = kernel!(QtipRaceV4L15AsPf1Sg8B32MetalKernel);
    let v4_l15as_sg16_b32 = kernel!(QtipRaceV4L15AsPf1Sg16B32MetalKernel);
    let v4_l15as_r2sg8_b32 = kernel!(QtipRaceV4L15AsR2Pf1Sg8B32MetalKernel);
    let v4_l15as_pf2sg8_b32 = kernel!(QtipRaceV4L15AsPf2Sg8B32MetalKernel);
    let v4_l15as_sg8_b64 = kernel!(QtipRaceV4L15AsPf1Sg8B64MetalKernel);
    let v4_l15as_sg16_b64 = kernel!(QtipRaceV4L15AsPf1Sg16B64MetalKernel);
    let v4_l15as_r2sg8_b64 = kernel!(QtipRaceV4L15AsR2Pf1Sg8B64MetalKernel);
    let v4_l15as_pf2sg8_b64 = kernel!(QtipRaceV4L15AsPf2Sg8B64MetalKernel);
    let v4_sw22_b32 = kernel!(QtipRaceV4L15Sw22B32MetalKernel);
    let v4_sw42_b32 = kernel!(QtipRaceV4L15Sw42B32MetalKernel);
    let v4_sw22s_b32 = kernel!(QtipRaceV4L15Sw22sB32MetalKernel);
    let v4_sw11_b32 = kernel!(QtipRaceV4L15Sw11B32MetalKernel);
    let v4_sw21_b32 = kernel!(QtipRaceV4L15Sw21B32MetalKernel);
    let v4_sw22_b64 = kernel!(QtipRaceV4L15Sw22B64MetalKernel);
    let v4_sw42_b64 = kernel!(QtipRaceV4L15Sw42B64MetalKernel);
    let v4_sw22s_b64 = kernel!(QtipRaceV4L15Sw22sB64MetalKernel);
    let v4_sw11_b64 = kernel!(QtipRaceV4L15Sw11B64MetalKernel);
    let v4_sw21_b64 = kernel!(QtipRaceV4L15Sw21B64MetalKernel);
    let k3_sw22_b32 = kernel!(QtipRaceK3Sw22B32MetalKernel);
    let k3_sw42_b32 = kernel!(QtipRaceK3Sw42B32MetalKernel);
    let k3_sw21_b32 = kernel!(QtipRaceK3Sw21B32MetalKernel);
    let k3_sw22_b64 = kernel!(QtipRaceK3Sw22B64MetalKernel);
    let k3_sw42_b64 = kernel!(QtipRaceK3Sw42B64MetalKernel);
    let k3_sw21_b64 = kernel!(QtipRaceK3Sw21B64MetalKernel);
    let k2_sw22_b32 = kernel!(QtipRaceK2Sw22B32MetalKernel);
    let k2_sw42_b32 = kernel!(QtipRaceK2Sw42B32MetalKernel);
    let k2_sw22_b64 = kernel!(QtipRaceK2Sw22B64MetalKernel);
    let k2_sw42_b64 = kernel!(QtipRaceK2Sw42B64MetalKernel);
    let k2_pf0_sg4_b32 = kernel!(QtipRaceK2Pf0Sg4B32MetalKernel);
    let k2_pf2_sg4_b32 = kernel!(QtipRaceK2Pf2Sg4B32MetalKernel);
    let k2_pf2_sg2_b32 = kernel!(QtipRaceK2Pf2Sg2B32MetalKernel);
    let k2_pf2_sg2_b64 = kernel!(QtipRaceK2Pf2Sg2B64MetalKernel);
    let k3_pf0_sg4_b32 = kernel!(QtipRaceK3Pf0Sg4B32MetalKernel);
    let k3_pf2_sg4_b32 = kernel!(QtipRaceK3Pf2Sg4B32MetalKernel);
    let k3_pf2_sg2_b32 = kernel!(QtipRaceK3Pf2Sg2B32MetalKernel);
    let k3_pf2_sg4_b64 = kernel!(QtipRaceK3Pf2Sg4B64MetalKernel);
    let k3_pf2_sg2_b64 = kernel!(QtipRaceK3Pf2Sg2B64MetalKernel);
    let v4_r2_pf2_sg2_b32 = kernel!(QtipRaceV4R2Pf2Sg2B32MetalKernel);
    let v4_r2_pf2_sg4_b32 = kernel!(QtipRaceV4R2Pf2Sg4B32MetalKernel);
    let k2_r2_pf2_sg2_b32 = kernel!(QtipRaceK2R2Pf2Sg2B32MetalKernel);
    let k2_r2_pf0_sg2_b32 = kernel!(QtipRaceK2R2Pf0Sg2B32MetalKernel);
    let k3_r2_pf2_sg2_b32 = kernel!(QtipRaceK3R2Pf2Sg2B32MetalKernel);
    let k3_r2_pf0_sg2_b32 = kernel!(QtipRaceK3R2Pf0Sg2B32MetalKernel);
    let v4_t2_pf2_b16 = kernel!(QtipRaceV4T2Pf2Sg2B16MetalKernel);
    let k3_t2_pf0_b16 = kernel!(QtipRaceK3T2Pf0Sg2B16MetalKernel);
    let k3_t4_pf0_b16 = kernel!(QtipRaceK3T4Pf0Sg2B16MetalKernel);
    let k2_t2_pf0_b16 = kernel!(QtipRaceK2T2Pf0Sg2B16MetalKernel);
    let k2_t2_pf2_b16 = kernel!(QtipRaceK2T2Pf2Sg2B16MetalKernel);
    let cs_pf2_sg4_b32 = (kernel!(QtipRaceV4CsPf2Sg4B32Pass0MetalKernel), kernel!(QtipRaceV4CsPf2Sg4B32Pass1MetalKernel));
    let cs_r2_pf2_sg2_b32 = (kernel!(QtipRaceV4CsR2Pf2Sg2B32Pass0MetalKernel), kernel!(QtipRaceV4CsR2Pf2Sg2B32Pass1MetalKernel));
    let cs_pf2_sg2_b64 = (kernel!(QtipRaceV4CsPf2Sg2B64Pass0MetalKernel), kernel!(QtipRaceV4CsPf2Sg2B64Pass1MetalKernel));
    let cs_t2_pf0_b16 = (kernel!(QtipRaceV4CsT2Pf0Sg2B16Pass0MetalKernel), kernel!(QtipRaceV4CsT2Pf0Sg2B16Pass1MetalKernel));
    let v4_as_pf1_sg4_b32 = kernel!(QtipRaceV4AsPf1Sg4B32MetalKernel);
    let v4_as_pf1_sg8_b32 = kernel!(QtipRaceV4AsPf1Sg8B32MetalKernel);
    let v4_as_pf1_sg16_b32 = kernel!(QtipRaceV4AsPf1Sg16B32MetalKernel);
    let v4_as_pf0_sg8_b32 = kernel!(QtipRaceV4AsPf0Sg8B32MetalKernel);
    let v4_as_r2_pf1_sg4_b32 = kernel!(QtipRaceV4AsR2Pf1Sg4B32MetalKernel);
    let v4_as_pf1_sg4_b64 = kernel!(QtipRaceV4AsPf1Sg4B64MetalKernel);
    let v4_as_pf1_sg8_b64 = kernel!(QtipRaceV4AsPf1Sg8B64MetalKernel);
    let v4_as_pf1_sg16_b64 = kernel!(QtipRaceV4AsPf1Sg16B64MetalKernel);
    let v4_as_r2_pf1_sg4_b64 = kernel!(QtipRaceV4AsR2Pf1Sg4B64MetalKernel);
    let v4_ascs_sg8_b32 = (kernel!(QtipRaceV4AsCsPf1Sg8B32Pass1MetalKernel), kernel!(QtipRaceV4AsCsPf1Sg8B32Pass2MetalKernel));
    let v4_ascs_sg16_b32 = (kernel!(QtipRaceV4AsCsPf1Sg16B32Pass1MetalKernel), kernel!(QtipRaceV4AsCsPf1Sg16B32Pass2MetalKernel));
    let v4_ascs_sg8_b64 = (kernel!(QtipRaceV4AsCsPf1Sg8B64Pass1MetalKernel), kernel!(QtipRaceV4AsCsPf1Sg8B64Pass2MetalKernel));
    let k3_as_pf1_sg8_b32 = kernel!(QtipRaceK3AsPf1Sg8B32MetalKernel);
    let k3_as_pf1_sg16_b32 = kernel!(QtipRaceK3AsPf1Sg16B32MetalKernel);
    let k3_as_r2_pf1_sg4_b32 = kernel!(QtipRaceK3AsR2Pf1Sg4B32MetalKernel);
    let k3_as_pf1_sg8_b64 = kernel!(QtipRaceK3AsPf1Sg8B64MetalKernel);
    let k3_as_pf1_sg16_b64 = kernel!(QtipRaceK3AsPf1Sg16B64MetalKernel);
    let k2_as_pf1_sg8_b32 = kernel!(QtipRaceK2AsPf1Sg8B32MetalKernel);
    let k2_as_pf1_sg8_b64 = kernel!(QtipRaceK2AsPf1Sg8B64MetalKernel);

    let variants = |geometry: Geometry, batch: u32| -> Vec<(&'static str, Runner<'_>)> {
        match (geometry, batch) {
            (Geometry::V4, 8 | 16) => vec![
                ("base", runner_base16!(base_v4_b16, transpose)),
                ("b32_pf2", runner!(v4_pf2_sg4_b32)),
                ("b32_pf2_sg2", runner!(v4_pf2_sg2_b32)),
                ("b32r2_pf2_sg2", runner!(v4_r2_pf2_sg2_b32)),
                ("b32r2_pf2_sg4", runner!(v4_r2_pf2_sg4_b32)),
                ("t2_pf2", runner!(v4_t2_pf2_b16)),
                ("sign12_t2", runner!(v4_sign12_t2_b16)),
                ("sign12_sg2_b32", runner!(v4_sign12_sg2_b32)),
                ("sign12_sg4_b32", runner!(v4_sign12_sg4_b32)),
                ("s14_t2", runner!(v4_s14_t2_b16)),
                ("anti_t2", runner!(v4_anti_t2_b16)),
                ("s14_sg4_b32", runner!(v4_sign14_sg4_b32)),
                ("s14_sg2_b32", runner!(v4_sign14_sg2_b32)),
                ("l15_t2_pf0", runner!(l15_t2_pf0_b16)),
                ("l15_t2_pf2", runner!(l15_t2_pf2_b16)),
                ("l15_t4_pf1", runner!(l15_t4_pf1_b16)),
                ("l15_sg4_b32", runner!(l15_pf2_sg4_b32)),
                ("l15_r2_b32", runner!(l15_r2_pf2_sg2_b32)),
                ("cs_t2_pf0", runner_cs!(cs_t2_pf0_b16.0, cs_t2_pf0_b16.1)),
                ("cs_b32_pf2", runner_cs!(cs_pf2_sg4_b32.0, cs_pf2_sg4_b32.1)),
                ("cs_b32r2_pf2", runner_cs!(cs_r2_pf2_sg2_b32.0, cs_r2_pf2_sg2_b32.1)),
            ],
            (Geometry::V4, 32) => vec![
                ("base", runner_base!(base_v4_b32)),
                ("pf2", runner!(v4_pf2_sg4_b32)),
                ("pf2_sg2", runner!(v4_pf2_sg2_b32)),
                ("r2_pf2_sg2", runner!(v4_r2_pf2_sg2_b32)),
                ("r2_pf2_sg4", runner!(v4_r2_pf2_sg4_b32)),
                ("as_pf1_sg4", runner!(v4_as_pf1_sg4_b32)),
                ("as_pf1_sg8", runner!(v4_as_pf1_sg8_b32)),
                ("as_pf1_sg16", runner!(v4_as_pf1_sg16_b32)),
                ("sign14_sg2", runner!(v4_sign14_sg2_b32)),
                ("sign14_sg4", runner!(v4_sign14_sg4_b32)),
                ("anti_sg2", runner!(v4_anti_sg2_b32)),
                ("anti_sg4", runner!(v4_anti_sg4_b32)),
                ("sign12_sg2", runner!(v4_sign12_sg2_b32)),
                ("sign12_sg4", runner!(v4_sign12_sg4_b32)),
                ("l15_as_sg8", runner!(v4_l15as_sg8_b32)),
                ("l15_as_sg16", runner!(v4_l15as_sg16_b32)),
                ("l15_as_r2sg8", runner!(v4_l15as_r2sg8_b32)),
                ("l15_as_pf2sg8", runner!(v4_l15as_pf2sg8_b32)),
                ("l15_sw22", runner!(v4_sw22_b32)),
                ("l15_sw42", runner!(v4_sw42_b32)),
                ("l15_sw22s", runner!(v4_sw22s_b32)),
                ("l15_sw11", runner!(v4_sw11_b32)),
                ("l15_sw21", runner!(v4_sw21_b32)),
                ("as_pf0_sg8", runner!(v4_as_pf0_sg8_b32)),
                ("as_r2_pf1_sg4", runner!(v4_as_r2_pf1_sg4_b32)),
                ("ascs_sg8", runner_cs!(v4_ascs_sg8_b32.0, v4_ascs_sg8_b32.1)),
                ("ascs_sg16", runner_cs!(v4_ascs_sg16_b32.0, v4_ascs_sg16_b32.1)),
                ("cs_pf2", runner_cs!(cs_pf2_sg4_b32.0, cs_pf2_sg4_b32.1)),
                ("cs_r2_pf2_sg2", runner_cs!(cs_r2_pf2_sg2_b32.0, cs_r2_pf2_sg2_b32.1)),
                ("l15_sg4", runner!(l15_pf2_sg4_b32)),
                ("l15_sg2", runner!(l15_pf2_sg2_b32)),
                ("l15_r2", runner!(l15_r2_pf2_sg2_b32)),
            ],
            (Geometry::V4, 64) => vec![
                ("base", runner_base!(base_v4_b64)),
                ("pf2", runner!(v4_pf2_sg4_b64)),
                ("pf2_sg2", runner!(v4_pf2_sg2_b64)),
                ("pf2_sg8", runner!(v4_pf2_sg8_b64)),
                ("cs_pf2_sg2", runner_cs!(cs_pf2_sg2_b64.0, cs_pf2_sg2_b64.1)),
                ("as_pf1_sg4", runner!(v4_as_pf1_sg4_b64)),
                ("as_pf1_sg8", runner!(v4_as_pf1_sg8_b64)),
                ("as_pf1_sg16", runner!(v4_as_pf1_sg16_b64)),
                ("s14_as_sg16", runner!(v4_s14_as_sg16_b64)),
                ("anti_as_sg16", runner!(v4_anti_as_sg16_b64)),
                ("sign14_sg2", runner!(v4_sign14_sg2_b64)),
                ("sign14_sg4", runner!(v4_sign14_sg4_b64)),
                ("anti_sg2", runner!(v4_anti_sg2_b64)),
                ("anti_sg4", runner!(v4_anti_sg4_b64)),
                ("sign12_sg2", runner!(v4_sign12_sg2_b64)),
                ("sign12_sg4", runner!(v4_sign12_sg4_b64)),
                ("l15_as_sg8", runner!(v4_l15as_sg8_b64)),
                ("l15_as_sg16", runner!(v4_l15as_sg16_b64)),
                ("l15_as_r2sg8", runner!(v4_l15as_r2sg8_b64)),
                ("l15_as_pf2sg8", runner!(v4_l15as_pf2sg8_b64)),
                ("l15_sw22", runner!(v4_sw22_b64)),
                ("l15_sw42", runner!(v4_sw42_b64)),
                ("l15_sw22s", runner!(v4_sw22s_b64)),
                ("l15_sw11", runner!(v4_sw11_b64)),
                ("l15_sw21", runner!(v4_sw21_b64)),
                ("as_r2_pf1_sg4", runner!(v4_as_r2_pf1_sg4_b64)),
                ("ascs_sg8", runner_cs!(v4_ascs_sg8_b64.0, v4_ascs_sg8_b64.1)),
                ("l15_sg2", runner!(l15_pf2_sg2_b64)),
                ("l15_sg4", runner!(l15_pf2_sg4_b64)),
                ("l15_sg8", runner!(l15_pf2_sg8_b64)),
            ],
            (Geometry::V2K2, 8 | 16) => vec![
                ("base", runner_base16!(base_k2_b16, transpose)),
                ("b32_pf2", runner!(k2_pf2_sg4_b32)),
                ("b32_pf2_sg2", runner!(k2_pf2_sg2_b32)),
                ("b32r2_pf2_sg2", runner!(k2_r2_pf2_sg2_b32)),
                ("b32r2_pf0_sg2", runner!(k2_r2_pf0_sg2_b32)),
                ("t2_pf0", runner!(k2_t2_pf0_b16)),
                ("t2_pf2", runner!(k2_t2_pf2_b16)),
                ("l15_t2", runner!(k2_l15_t2_b16)),
                ("l15_t2pf2", runner!(k2_l15_t2pf2_b16)),
            ],
            (Geometry::V2K2, 32) => vec![
                ("base", runner_base!(base_k2_b32)),
                ("pf0", runner!(k2_pf0_sg4_b32)),
                ("pf2", runner!(k2_pf2_sg4_b32)),
                ("pf2_sg2", runner!(k2_pf2_sg2_b32)),
                ("r2_pf2_sg2", runner!(k2_r2_pf2_sg2_b32)),
                ("as_pf1_sg8", runner!(k2_as_pf1_sg8_b32)),
                ("l15_sg2", runner!(k2_l15_sg2_b32)),
                ("l15_pf0sg4", runner!(k2_l15_pf0sg4_b32)),
                ("l15_r2", runner!(k2_l15_r2_b32)),
                ("sw22", runner!(k2_sw22_b32)),
                ("sw42", runner!(k2_sw42_b32)),
            ],
            (Geometry::V2K2, 64) => vec![
                ("base", runner_base!(base_k2_b64)),
                ("pf2_sg2", runner!(k2_pf2_sg2_b64)),
                ("as_pf1_sg8", runner!(k2_as_pf1_sg8_b64)),
                ("l15_sg2", runner!(k2_l15_sg2_b64)),
                ("sw22", runner!(k2_sw22_b64)),
                ("sw42", runner!(k2_sw42_b64)),
            ],
            (Geometry::V2K3, 8 | 16) => vec![
                ("base", runner_base16!(base_k3_b16, transpose)),
                ("b32_pf2", runner!(k3_pf2_sg4_b32)),
                ("b32_pf2_sg2", runner!(k3_pf2_sg2_b32)),
                ("b32r2_pf2_sg2", runner!(k3_r2_pf2_sg2_b32)),
                ("b32r2_pf0_sg2", runner!(k3_r2_pf0_sg2_b32)),
                ("t2_pf0", runner!(k3_t2_pf0_b16)),
                ("l15_t2", runner!(k3_l15_t2_b16)),
                ("l15_t4", runner!(k3_l15_t4_b16)),
                ("t4_pf0", runner!(k3_t4_pf0_b16)),
            ],
            (Geometry::V2K3, 32) => vec![
                ("base", runner_base!(base_k3_b32)),
                ("pf0", runner!(k3_pf0_sg4_b32)),
                ("pf2", runner!(k3_pf2_sg4_b32)),
                ("pf2_sg2", runner!(k3_pf2_sg2_b32)),
                ("r2_pf2_sg2", runner!(k3_r2_pf2_sg2_b32)),
                ("as_pf1_sg8", runner!(k3_as_pf1_sg8_b32)),
                ("as_pf1_sg16", runner!(k3_as_pf1_sg16_b32)),
                ("l15_sg2", runner!(k3_l15_sg2_b32)),
                ("l15_pf0sg4", runner!(k3_l15_pf0sg4_b32)),
                ("l15_r2", runner!(k3_l15_r2_b32)),
                ("sw22", runner!(k3_sw22_b32)),
                ("sw42", runner!(k3_sw42_b32)),
                ("sw21", runner!(k3_sw21_b32)),
                ("as_r2_pf1_sg4", runner!(k3_as_r2_pf1_sg4_b32)),
            ],
            (Geometry::V2K3, 64) => vec![
                ("base", runner_base!(base_k3_b64)),
                ("pf2", runner!(k3_pf2_sg4_b64)),
                ("pf2_sg2", runner!(k3_pf2_sg2_b64)),
                ("as_pf1_sg8", runner!(k3_as_pf1_sg8_b64)),
                ("as_pf1_sg16", runner!(k3_as_pf1_sg16_b64)),
                ("l15_sg4", runner!(k3_l15_sg4_b64)),
                ("l15_sg2", runner!(k3_l15_sg2_b64)),
                ("sw22", runner!(k3_sw22_b64)),
                ("sw42", runner!(k3_sw42_b64)),
                ("sw21", runner!(k3_sw21_b64)),
            ],
            _ => panic!("unsupported batch {batch}"),
        }
    };

    let mut g64_kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        DataType::BF16,
        DataType::BF16,
        DataType::BF16,
    )
    .expect("G64 kernel");

    let mut mismatches = 0usize;
    for &batch in &batches {
        let padded_batch = batch.max(32);
        let mut total_base = 0.0f64;
        let mut total_g64 = 0.0f64;
        let mut total_best = 0.0f64;
        let mut totals_by_variant: Vec<(&'static str, f64)> = Vec::new();
        println!("\n=== batch {batch} (activations padded to {padded_batch}, inner {inner}, reps {reps}) ===");
        let mut families: Vec<Family> = Vec::new();
        for family in FAMILIES.iter() {
            if let Some(filter) = &family_filter
                && !format!("{}:{:?}", family.name, family.geometry).contains(filter.as_str())
            {
                continue;
            }
            if rows_override.is_empty() {
                families.push(Family { ..*family });
            } else {
                for &rows in &rows_override {
                    families.push(Family { rows, leaves: 1, ..*family });
                }
            }
        }
        for (family_index, family) in families.iter().enumerate() {
            let case = build_case(&context, family, batch, padded_batch, 1000 + family_index as u64);
            let mut output = alloc_allocation::<Metal, bf16>(&context, family.rows as usize * batch as usize);
            let mut scratch = alloc_allocation::<Metal, i32>(&context, family.rows as usize * 64);
            let mut reference: Option<Vec<bf16>> = None;
            let oracle = oracle_rows.map(|limit| cpu_reference(&case, limit));
            let mut line = format!(
                "{:<9} {:<5} {:>6}x{:<6} x{:<3}",
                family.name,
                format!("{:?}", family.geometry),
                family.rows,
                family.columns,
                family.leaves
            );
            let mut best = f64::INFINITY;
            let mut base_time = 0.0f64;
            for (variant_name, runner) in variants(family.geometry, batch) {
                let is_diag = variant_name.starts_with("diag_");
                if is_diag && skip_diag {
                    continue;
                }
                if variant_name != "base"
                    && let Some(filter) = &variant_filter
                    && !filter.split(',').any(|item| item.trim() == variant_name)
                {
                    continue;
                }
                let mut times = Vec::with_capacity(reps);
                for _ in 0..reps + 1 {
                    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
                    for _ in 0..inner {
                        runner(&case, &mut output, &mut scratch, &mut encoder);
                    }
                    let seconds = encoder
                        .end_encoding()
                        .submit()
                        .wait_until_completed()
                        .expect("execution")
                        .gpu_execution_time()
                        .as_secs_f64();
                    times.push(seconds / inner as f64);
                }
                times.remove(0);
                let seconds = median(times);
                if !is_diag {
                    let values = allocation_to_vec::<Metal, bf16>(&output);
                    if let (Some(oracle), Some(limit)) = (&oracle, oracle_rows) {
                        let rows = family.rows as usize;
                        let bad = (0..batch as usize)
                            .flat_map(|token| (0..limit.min(rows)).map(move |row| token * rows + row))
                            .filter(|&index| oracle[index].to_bits() != values[index].to_bits())
                            .count();
                        line.push_str(&format!(" <{variant_name} cpu_bad={bad}>"));
                    }
                    match &reference {
                        None => reference = Some(values),
                        Some(expected) => {
                            let bad =
                                expected.iter().zip(&values).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                            if bad != 0 {
                                mismatches += 1;
                                line.push_str(&format!(" [{variant_name} MISMATCH {bad}]"));
                                if std::env::var("QTIP_RACE_DEBUG").is_ok() {
                                    let rows = family.rows as usize;
                                    let mut by_token = vec![0usize; batch as usize];
                                    let mut by_row_bucket = vec![0usize; 8];
                                    let mut first_rows: Vec<usize> = Vec::new();
                                    for (index, (a, b)) in expected.iter().zip(&values).enumerate() {
                                        if a.to_bits() != b.to_bits() {
                                            by_token[index / rows] += 1;
                                            by_row_bucket[((index % rows) % 64) / 8] += 1;
                                            if first_rows.len() < 12 {
                                                first_rows.push(index % rows);
                                            }
                                        }
                                    }
                                    println!("    debug {variant_name}: mismatches by token {by_token:?}; by (row%64)/8 bucket {by_row_bucket:?}; first rows {first_rows:?}");
                                    let first = expected.iter().zip(&values).enumerate().find(|(_, (a, b))| a.to_bits() != b.to_bits());
                                    let sample: Vec<String> = (0..6).map(|i| format!("{}/{}", expected[i], values[i])).collect();
                                    println!("    debug {variant_name}: first mismatch at {:?}; expected/got[0..6] = {:?}", first.map(|(i, _)| i), sample);
                                }
                            }
                        },
                    }
                }
                if variant_name == "base" {
                    base_time = seconds;
                } else if !is_diag {
                    best = best.min(seconds);
                }
                if variant_name != "base" {
                    match totals_by_variant.iter_mut().find(|(name, _)| *name == variant_name) {
                        Some((_, total)) => *total += seconds * family.leaves as f64,
                        None => totals_by_variant.push((variant_name, seconds * family.leaves as f64)),
                    }
                }
                line.push_str(&format!(" {variant_name}={:.1}", seconds * 1e6));
            }
            if best.is_infinite() {
                best = base_time;
            }
            let g64_time = if skip_g64 {
                0.0
            } else {
                let input = QuantInput::<bf16>::new(
                    batch,
                    family.columns,
                    family.rows,
                    64,
                    4,
                    QuantizationMethod::ScaleBias,
                    77 + family_index as u64,
                );
                let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
                let mut times = Vec::with_capacity(reps);
                for _ in 0..reps + 1 {
                    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
                    for _ in 0..inner {
                        g64_kernel.encode(quant_arguments(&mut buffers, &input), &mut encoder).expect("encode G64");
                    }
                    times.push(
                        encoder
                            .end_encoding()
                            .submit()
                            .wait_until_completed()
                            .expect("G64 execution")
                            .gpu_execution_time()
                            .as_secs_f64()
                            / inner as f64,
                    );
                }
                times.remove(0);
                median(times)
            };
            line.push_str(&format!(
                " | g64={:.1}  best/g64={:.3} base/g64={:.3}",
                g64_time * 1e6,
                best / g64_time,
                base_time / g64_time
            ));
            println!("{line}");
            total_base += base_time * family.leaves as f64;
            total_best += best * family.leaves as f64;
            total_g64 += g64_time * family.leaves as f64;
        }
        println!(
            "TOTAL batch {batch}: base={:.3}ms best={:.3}ms g64={:.3}ms  base/g64={:.3} best/g64={:.3}",
            total_base * 1e3,
            total_best * 1e3,
            total_g64 * 1e3,
            total_base / total_g64,
            total_best / total_g64
        );
        for (name, total) in &totals_by_variant {
            println!("   variant {name:<16} total={:.3}ms  /g64={:.3}", total * 1e3, total / total_g64);
        }
    }
    if oracle_rows.is_none() {
        assert_eq!(mismatches, 0, "race kernels must be bit-exact against the physical kernels");
    } else {
        println!("base-mismatch count (informational with the CPU oracle active): {mismatches}");
    }
}

/// The weaver's sparse i3/S4 readout must reproduce the dense readout on the gathered columns.
#[uzu_test]
fn qtip_i3_sparse_readout_matches_dense() {
    use crate::backends::common::{
        Context,
        kernel::qtip_s_exact::{I3S4ReadoutArguments, I3S4SparseReadoutArguments, QtipSExactKernel},
    };
    let context = crate::tests::util::shared_metal_context();
    assert!(context.supports_mxu(), "i3 readout requires MXU support");
    let kernel = <<Metal as Backend>::Kernels as Kernels>::QtipSExactKernel::new(&context).expect("kernel");
    let mut rng = SmallRng::seed_from_u64(7);
    let (vocab, model_dim, rows, ids_per_row) = (4096u32, 5120u32, 5u32, 512u32);
    let codes: Vec<u8> = (0..vocab as usize * model_dim as usize * 3 / 8).map(|_| rng.random::<u8>()).collect();
    let ladder_indices: Vec<u8> = (0..vocab as usize * model_dim as usize / 128).map(|_| rng.random::<u8>()).collect();
    let ladder: Vec<f16> = (0..16).map(|i| f16::from_f32(0.02 + 0.01 * i as f32)).collect();
    let row_scales: Vec<bf16> = (0..vocab).map(|_| bf16::from_f32(rng.random_range(0.5f32..2.0))).collect();
    let factors: Vec<i32> = (0..model_dim).map(|_| if rng.random::<bool>() { 1 } else { -1 }).collect();
    let input: Vec<bf16> =
        (0..rows as usize * model_dim as usize).map(|_| bf16::from_f32(rng.random_range(-1.0f32..1.0))).collect();
    let token_ids: Vec<u32> = (0..rows as usize * ids_per_row as usize).map(|_| rng.random_range(0..vocab)).collect();

    let codes_a = alloc_allocation_with_data::<Metal, u8>(&context, &codes);
    let ladder_indices_a = alloc_allocation_with_data::<Metal, u8>(&context, &ladder_indices);
    let ladder_a = alloc_allocation_with_data::<Metal, f16>(&context, &ladder);
    let row_scales_a = alloc_allocation_with_data::<Metal, bf16>(&context, &row_scales);
    let factors_a = alloc_allocation_with_data::<Metal, i32>(&context, &factors);
    let input_a = alloc_allocation_with_data::<Metal, bf16>(&context, &input);
    let token_ids_a = alloc_allocation_with_data::<Metal, u32>(&context, &token_ids);

    // own the scratch pool so the kernel outputs stay alive until they are read back
    let pool = std::sync::Arc::new(context.create_allocation_pool(false));
    let mut encoder =
        Encoder::<Metal>::new_with_pool_name(&context, pool.clone(), Some("sparse i3 readout")).expect("encoder");
    let dense = kernel
        .encode_i3_s4_readout(
            I3S4ReadoutArguments {
                input: &input_a,
                codes: &codes_a,
                row_scales: &row_scales_a,
                ladder_indices: &ladder_indices_a,
                ladder: &ladder_a,
                input_hadamard_factors: &factors_a,
                batch: rows,
                vocab_size: vocab,
                model_dim,
                output_data_type: DataType::BF16,
            },
            &mut encoder,
        )
        .expect("dense readout");
    let sparse = kernel
        .encode_i3_s4_readout_sparse(
            I3S4SparseReadoutArguments {
                input: &input_a,
                token_ids: &token_ids_a,
                codes: &codes_a,
                row_scales: &row_scales_a,
                ladder_indices: &ladder_indices_a,
                ladder: &ladder_a,
                input_hadamard_factors: &factors_a,
                rows,
                ids_per_row,
                vocab_size: vocab,
                model_dim,
                output_data_type: DataType::BF16,
                soft_cap: 0.0,
            },
            &mut encoder,
        )
        .expect("sparse readout");
    encoder.end_encoding().submit().wait_until_completed().expect("execution");
    let (dense, sparse) = (allocation_to_vec::<Metal, bf16>(&dense), allocation_to_vec::<Metal, bf16>(&sparse));
    let mut max_abs = 0f32;
    let mut max_ref = 0f32;
    let mut sum_sq_err = 0f64;
    let mut sum_sq_ref = 0f64;
    for r in 0..rows as usize {
        for j in 0..ids_per_row as usize {
            let token = token_ids[r * ids_per_row as usize + j] as usize;
            let reference = dense[r * vocab as usize + token].to_f32();
            let value = sparse[r * ids_per_row as usize + j].to_f32();
            max_abs = max_abs.max((reference - value).abs());
            max_ref = max_ref.max(reference.abs());
            sum_sq_err += ((reference - value) as f64).powi(2);
            sum_sq_ref += (reference as f64).powi(2);
        }
    }
    let rel_rms = (sum_sq_err / sum_sq_ref).sqrt();
    println!("sparse-vs-dense i3 readout: max_abs={max_abs:.4e} max_ref={max_ref:.4e} rel_rms={rel_rms:.3e}");
    assert!(max_ref > 0.0, "degenerate reference");
    assert!(rel_rms < 1e-2, "sparse i3 readout deviates from dense: rel_rms={rel_rms}");
    assert!(max_abs < 2e-2 * max_ref, "sparse i3 readout max deviation too large: {max_abs} vs {max_ref}");
}

/// The tiered repack must decode every band onto the signed-nibble GEMM grid exactly as a reference decode does.
#[uzu_test]
fn qtip_tiered_head_repack_matches_reference() {
    use crate::encodable_block::embedding::{repack_i3_s4_to_symmetric_gemm, repack_tiered_to_symmetric_gemm};
    let mut rng = SmallRng::seed_from_u64(11);
    let (vocab, dim, hot, cold) = (96usize, 256usize, 32usize, 64usize);
    let codes4: Vec<u8> = (0..hot * dim / 2).map(|_| rng.random::<u8>()).collect();
    let codes3: Vec<u8> = (0..(cold - hot) * dim * 3 / 8).map(|_| rng.random::<u8>()).collect();
    let codes2: Vec<u8> = (0..(vocab - cold) * dim / 4).map(|_| rng.random::<u8>()).collect();
    let row_scales: Vec<bf16> = (0..vocab).map(|_| bf16::from_f32(rng.random_range(0.5f32..2.0))).collect();
    let ladder_indices: Vec<u8> = (0..vocab * dim / 128).map(|_| rng.random::<u8>()).collect();
    let ladder: Vec<f16> = (0..16).map(|i| f16::from_f32(2f32.powf((i as f32 - 11.0) / 2.0))).collect();
    let (codes, scales) =
        repack_tiered_to_symmetric_gemm(hot, cold, &codes4, &codes3, &codes2, &row_scales, &ladder_indices, &ladder, vocab, dim, false);
    assert_eq!(codes.len(), vocab * dim / 2);
    assert_eq!(scales.len(), vocab * dim / 64);
    // reference decode
    for row in 0..vocab {
        for column in 0..dim {
            let expected: i32 = if row < hot {
                let byte = codes4[row * dim / 2 + column / 2];
                (if column % 2 == 0 { byte & 15 } else { byte >> 4 }) as i32 - 8
            } else if row < cold {
                let local = row - hot;
                let bit = column * 3;
                let src = &codes3[local * dim * 3 / 8..(local + 1) * dim * 3 / 8];
                let mut packed = src[bit >> 3] as u32;
                if (bit & 7) > 5 {
                    packed |= (src[(bit >> 3) + 1] as u32) << 8;
                }
                (((packed >> (bit & 7)) & 7) * 2) as i32 - 7
            } else {
                let local = row - cold;
                let byte = codes2[local * dim / 4 + column / 4];
                (((byte >> (2 * (column % 4))) & 3) as i32) * 2 - 3
            };
            let nibble = codes[row * dim / 2 + column / 2];
            let nibble = if column % 2 == 0 { nibble & 15 } else { nibble >> 4 };
            let got = ((nibble as i8) << 4 >> 4) as i32; // sign-extend the nibble
            assert_eq!(got, expected, "row {row} column {column}");
        }
        for group in 0..dim / 64 {
            let packed = ladder_indices[row * dim / 128 + group / 2];
            let index = if group % 2 == 0 { packed & 15 } else { packed >> 4 } as usize;
            let expected = bf16::from_f32(row_scales[row].to_f32() * ladder[index].to_f32());
            assert_eq!(scales[row * dim / 64 + group], expected, "row {row} group {group}");
        }
    }
    // the plain i3 wrapper must agree with the tiered repack when every row is in the middle band
    let all3: Vec<u8> = (0..vocab * dim * 3 / 8).map(|_| rng.random::<u8>()).collect();
    let a = repack_i3_s4_to_symmetric_gemm(&all3, &row_scales, &ladder_indices, &ladder, vocab, dim, false);
    let b = repack_tiered_to_symmetric_gemm(0, vocab, &[], &all3, &[], &row_scales, &ladder_indices, &ladder, vocab, dim, false);
    assert_eq!(a.0, b.0);
    assert_eq!(a.1, b.1);
}
