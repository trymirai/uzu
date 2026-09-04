use crate::{
    array::size_for_shape,
    backends::{
        common::{Allocation, Encoder, kernel::qtip_s_exact::*},
        metal::{
            Metal, MetalContext,
            error::MetalError,
            kernel::{
                QtipD4S4EmbeddingLookupMetalKernel, QtipFullIncoherenceA8MetalKernel,
                QtipGaussianPhysicalQ8V2A8DirectK2B16MetalKernel, QtipGaussianPhysicalQ8V2A8DirectK2B32MetalKernel,
                QtipGaussianPhysicalQ8V2A8DirectK2B32BatchRowsMetalKernel,
                QtipGaussianPhysicalQ8V2A8DirectK2B64MetalKernel, QtipGaussianPhysicalQ8V2A8DirectK3B16MetalKernel,
                QtipGaussianPhysicalQ8V2A8DirectK2B64BatchRowsMetalKernel,
                QtipGaussianPhysicalQ8V2A8DirectK3B32MetalKernel, QtipGaussianPhysicalQ8V2A8DirectK3B64MetalKernel,
                QtipGaussianPhysicalQ8V2A8DirectK3B32BatchRowsMetalKernel,
                QtipGaussianPhysicalQ8V2A8DirectK3B64BatchRowsMetalKernel,
                QtipGaussianPhysicalQ8V4A8DirectB16MetalKernel, QtipGaussianPhysicalQ8V4A8DirectB32MetalKernel,
                QtipGaussianPhysicalQ8V4A8DirectB32BatchRowsMetalKernel,
                QtipGaussianPhysicalQ8V4A8DirectB64MetalKernel, QtipI3S4ReadoutMxuB16MetalKernel,
                QtipI3S4ReadoutSparseBf16MetalKernel, QtipI3S4ReadoutSparseF32MetalKernel, QtipResidualMergeHotMetalKernel,
                QtipGaussianPhysicalQ8V4A8DirectB64BatchRowsMetalKernel, QtipI3S4ReadoutMxuB32MetalKernel,
                QtipI3S4ReadoutMxuB64MetalKernel, QtipRaceK2Pf0Sg4B32MetalKernel, QtipRaceK2Pf2Sg2B32MetalKernel,
                QtipRaceK2Pf2Sg2B64MetalKernel, QtipRaceK2Pf2Sg4B32MetalKernel, QtipRaceK2R2Pf0Sg2B32MetalKernel,
                QtipRaceK2R2Pf2Sg2B32MetalKernel, QtipRaceK3Pf0Sg4B32MetalKernel, QtipRaceK3Pf2Sg2B32MetalKernel,
                QtipRaceK3Pf2Sg2B64MetalKernel, QtipRaceK3Pf2Sg4B32MetalKernel, QtipRaceK3Pf2Sg4B64MetalKernel,
                QtipRaceK3R2Pf0Sg2B32MetalKernel, QtipRaceK3R2Pf2Sg2B32MetalKernel, QtipRaceTransform5120MetalKernel,
                QtipRaceTransform6144MetalKernel, QtipRaceTransform17408MetalKernel, QtipRaceV4Pf2Sg2B32MetalKernel,
                QtipRaceV4Pf2Sg2B64MetalKernel, QtipRaceV4Pf2Sg4B32MetalKernel, QtipRaceV4Pf2Sg4B64MetalKernel,
                QtipRaceV4Pf2Sg8B64MetalKernel, QtipRaceV4R2Pf2Sg2B32MetalKernel, QtipRaceV4R2Pf2Sg4B32MetalKernel,
                QtipRaceV4T2Pf2Sg2B16MetalKernel, QtipRaceK3T4Pf0Sg2B16MetalKernel, QtipRaceK3T2Pf0Sg2B16MetalKernel,
                QtipRaceK2T2Pf0Sg2B16MetalKernel, QtipRaceK2T2Pf2Sg2B16MetalKernel, QtipRacePermuteHalvesMetalKernel,
                QtipRaceV4CsPf2Sg4B32Pass0MetalKernel, QtipRaceV4CsPf2Sg4B32Pass1MetalKernel,
                QtipRaceV4CsR2Pf2Sg2B32Pass0MetalKernel, QtipRaceV4CsR2Pf2Sg2B32Pass1MetalKernel,
                QtipRaceV4CsPf2Sg2B64Pass0MetalKernel, QtipRaceV4CsPf2Sg2B64Pass1MetalKernel,
                QtipRaceV4CsT2Pf0Sg2B16Pass0MetalKernel, QtipRaceV4CsT2Pf0Sg2B16Pass1MetalKernel,
                QtipRaceK3AsPf1Sg16B64MetalKernel, QtipRaceK2AsPf1Sg8B64MetalKernel,
                QtipRaceK3L15Pf2Sg2B32MetalKernel, QtipRaceK3L15Pf0Sg4B32MetalKernel, QtipRaceK3L15R2Pf2Sg2B32MetalKernel,
                QtipRaceK3L15Pf2Sg2B64MetalKernel, QtipRaceK3L15Pf2Sg4B64MetalKernel, QtipRaceK3L15T2Pf0Sg2B16MetalKernel,
                QtipRaceK3L15T4Pf0Sg2B16MetalKernel, QtipRaceK2L15Pf2Sg2B32MetalKernel, QtipRaceK2L15R2Pf2Sg2B32MetalKernel, QtipRaceK2L15Pf0Sg4B32MetalKernel,
                QtipRaceK2L15Pf2Sg2B64MetalKernel, QtipRaceK2L15T2Pf0Sg2B16MetalKernel, QtipRaceK2L15T2Pf2Sg2B16MetalKernel,
                QtipRaceV4Sign12Pf2Sg2B32MetalKernel, QtipRaceV4Sign12Pf2Sg4B32MetalKernel, QtipRaceV4Sign12Pf2Sg2B64MetalKernel,
                QtipRaceV4Sign12Pf2Sg4B64MetalKernel, QtipRaceV4Sign12T2Pf2Sg2B16MetalKernel,
                QtipRaceV4L15AsPf1Sg16B64MetalKernel,
                QtipRaceV4AntiPf2Sg2B32MetalKernel, QtipRaceV4AntiPf2Sg4B32MetalKernel, QtipRaceV4AntiPf2Sg2B64MetalKernel,
                QtipRaceV4AntiPf2Sg4B64MetalKernel, QtipRaceV4AntiT2Pf2Sg2B16MetalKernel, QtipRaceV4AntiAsPf1Sg16B64MetalKernel,
                QtipRaceV4Sign14Pf2Sg2B32MetalKernel, QtipRaceV4Sign14Pf2Sg4B32MetalKernel, QtipRaceV4Sign14Pf2Sg2B64MetalKernel,
                QtipRaceV4Sign14Pf2Sg4B64MetalKernel, QtipRaceV4Sign14T2Pf2Sg2B16MetalKernel, QtipRaceV4Sign14AsPf1Sg16B64MetalKernel,
                QtipRaceV4L15Pf2Sg4B32MetalKernel, QtipRaceV4L15Pf2Sg2B32MetalKernel, QtipRaceV4L15Pf2Sg2B64MetalKernel, QtipRaceV4L15Pf2Sg4B64MetalKernel, QtipRaceV4L15Pf2Sg8B64MetalKernel, QtipRaceV4L15R2Pf2Sg2B32MetalKernel, QtipRaceV4L15T2Pf0Sg2B16MetalKernel, QtipRaceV4L15T2Pf2Sg2B16MetalKernel, QtipRaceV4L15T4Pf1Sg2B16MetalKernel,
                QtipRht32Bf16PaddedMetalKernel, QtipRowsBatchToBatchRowsBf16MetalKernel,
                QtipRowsBatchToBatchRowsF32MetalKernel,
            },
        },
    },
    data_type::DataType,
};

/// Physical QTIP S dispatch.
///
/// Race configuration (env overrides for A/B benchmarking):
///   QTIP_RACE_TRANSFORM=0  use the original full-incoherence transform kernel
///   QTIP_RACE_PROJ=0       use the original physical projection kernels
///
/// The race projection path always runs 32- or 64-token MXU tiles (the
/// 16-token row-paired MXU path of the physical B16 kernels is numerically
/// wrong for half of the rows) and writes batch-rows output for exactly
/// `batch` tokens, so no transpose pass runs at any batch size. Kernel choices
/// per (codec, shape, batch) come from the CPU-oracle-validated race profile.
pub struct QtipSExactMetalKernel {
    race_transform: bool,
    race_projection: bool,
    /// timing probes: skip the projection / transform kernels (outputs are garbage)
    null_projection: bool,
    null_transform: bool,
    full_incoherence_a8: QtipFullIncoherenceA8MetalKernel,
    transform_5120: QtipRaceTransform5120MetalKernel,
    transform_6144: QtipRaceTransform6144MetalKernel,
    transform_17408: QtipRaceTransform17408MetalKernel,
    v2_k2_b16: QtipGaussianPhysicalQ8V2A8DirectK2B16MetalKernel,
    v2_k2_b32: QtipGaussianPhysicalQ8V2A8DirectK2B32MetalKernel,
    v2_k2_b32_batch_rows: QtipGaussianPhysicalQ8V2A8DirectK2B32BatchRowsMetalKernel,
    v2_k2_b64: QtipGaussianPhysicalQ8V2A8DirectK2B64MetalKernel,
    v2_k2_b64_batch_rows: QtipGaussianPhysicalQ8V2A8DirectK2B64BatchRowsMetalKernel,
    v2_k3_b16: QtipGaussianPhysicalQ8V2A8DirectK3B16MetalKernel,
    v2_k3_b32: QtipGaussianPhysicalQ8V2A8DirectK3B32MetalKernel,
    v2_k3_b32_batch_rows: QtipGaussianPhysicalQ8V2A8DirectK3B32BatchRowsMetalKernel,
    v2_k3_b64: QtipGaussianPhysicalQ8V2A8DirectK3B64MetalKernel,
    v2_k3_b64_batch_rows: QtipGaussianPhysicalQ8V2A8DirectK3B64BatchRowsMetalKernel,
    v4_b16: QtipGaussianPhysicalQ8V4A8DirectB16MetalKernel,
    v4_b32: QtipGaussianPhysicalQ8V4A8DirectB32MetalKernel,
    v4_b32_batch_rows: QtipGaussianPhysicalQ8V4A8DirectB32BatchRowsMetalKernel,
    v4_b64: QtipGaussianPhysicalQ8V4A8DirectB64MetalKernel,
    v4_b64_batch_rows: QtipGaussianPhysicalQ8V4A8DirectB64BatchRowsMetalKernel,
    // race kernels (all take `active_batch`, write [batch, rows])
    v4_pf2_sg2_b32: QtipRaceV4Pf2Sg2B32MetalKernel,
    v4_pf2_sg4_b32: QtipRaceV4Pf2Sg4B32MetalKernel,
    v4_r2_pf2_sg2_b32: QtipRaceV4R2Pf2Sg2B32MetalKernel,
    v4_r2_pf2_sg4_b32: QtipRaceV4R2Pf2Sg4B32MetalKernel,
    v4_pf2_sg2_b64: QtipRaceV4Pf2Sg2B64MetalKernel,
    v4_pf2_sg4_b64: QtipRaceV4Pf2Sg4B64MetalKernel,
    v4_pf2_sg8_b64: QtipRaceV4Pf2Sg8B64MetalKernel,
    k3_pf0_sg4_b32: QtipRaceK3Pf0Sg4B32MetalKernel,
    k3_pf2_sg2_b32: QtipRaceK3Pf2Sg2B32MetalKernel,
    k3_pf2_sg4_b32: QtipRaceK3Pf2Sg4B32MetalKernel,
    k3_r2_pf0_sg2_b32: QtipRaceK3R2Pf0Sg2B32MetalKernel,
    k3_r2_pf2_sg2_b32: QtipRaceK3R2Pf2Sg2B32MetalKernel,
    k3_pf2_sg2_b64: QtipRaceK3Pf2Sg2B64MetalKernel,
    k3_pf2_sg4_b64: QtipRaceK3Pf2Sg4B64MetalKernel,
    k2_pf0_sg4_b32: QtipRaceK2Pf0Sg4B32MetalKernel,
    k2_pf2_sg2_b32: QtipRaceK2Pf2Sg2B32MetalKernel,
    k2_pf2_sg4_b32: QtipRaceK2Pf2Sg4B32MetalKernel,
    k2_r2_pf0_sg2_b32: QtipRaceK2R2Pf0Sg2B32MetalKernel,
    k2_r2_pf2_sg2_b32: QtipRaceK2R2Pf2Sg2B32MetalKernel,
    k2_pf2_sg2_b64: QtipRaceK2Pf2Sg2B64MetalKernel,
    // component split (V4): two half-tables, K-packed activation halves, int32 partials
    // L=15 (32768-state) V4 kernels: single pass, masked states
    l15_pf2_sg4_b32: QtipRaceV4L15Pf2Sg4B32MetalKernel,
    l15_pf2_sg2_b32: QtipRaceV4L15Pf2Sg2B32MetalKernel,
    l15_r2_pf2_sg2_b32: QtipRaceV4L15R2Pf2Sg2B32MetalKernel,
    l15_pf2_sg2_b64: QtipRaceV4L15Pf2Sg2B64MetalKernel,
    l15_pf2_sg4_b64: QtipRaceV4L15Pf2Sg4B64MetalKernel,
    l15_pf2_sg8_b64: QtipRaceV4L15Pf2Sg8B64MetalKernel,
    l15_t2_pf0_b16: QtipRaceV4L15T2Pf0Sg2B16MetalKernel,
    l15_t2_pf2_b16: QtipRaceV4L15T2Pf2Sg2B16MetalKernel,
    l15_t4_pf1_b16: QtipRaceV4L15T4Pf1Sg2B16MetalKernel,
    l15_as_sg16_b64: QtipRaceV4L15AsPf1Sg16B64MetalKernel,
    anti_pf2_sg2_b32: QtipRaceV4AntiPf2Sg2B32MetalKernel,
    anti_pf2_sg4_b32: QtipRaceV4AntiPf2Sg4B32MetalKernel,
    anti_pf2_sg2_b64: QtipRaceV4AntiPf2Sg2B64MetalKernel,
    anti_pf2_sg4_b64: QtipRaceV4AntiPf2Sg4B64MetalKernel,
    anti_t2_pf2_b16: QtipRaceV4AntiT2Pf2Sg2B16MetalKernel,
    anti_as_sg16_b64: QtipRaceV4AntiAsPf1Sg16B64MetalKernel,
    s14_pf2_sg2_b32: QtipRaceV4Sign14Pf2Sg2B32MetalKernel,
    s14_pf2_sg4_b32: QtipRaceV4Sign14Pf2Sg4B32MetalKernel,
    s14_pf2_sg2_b64: QtipRaceV4Sign14Pf2Sg2B64MetalKernel,
    s14_pf2_sg4_b64: QtipRaceV4Sign14Pf2Sg4B64MetalKernel,
    s14_t2_pf2_b16: QtipRaceV4Sign14T2Pf2Sg2B16MetalKernel,
    s14_as_sg16_b64: QtipRaceV4Sign14AsPf1Sg16B64MetalKernel,
    k3_l15_sg2_b32: QtipRaceK3L15Pf2Sg2B32MetalKernel,
    k3_l15_pf0sg4_b32: QtipRaceK3L15Pf0Sg4B32MetalKernel,
    k3_l15_r2_b32: QtipRaceK3L15R2Pf2Sg2B32MetalKernel,
    k3_l15_sg2_b64: QtipRaceK3L15Pf2Sg2B64MetalKernel,
    k3_l15_sg4_b64: QtipRaceK3L15Pf2Sg4B64MetalKernel,
    k3_l15_t2_b16: QtipRaceK3L15T2Pf0Sg2B16MetalKernel,
    k3_l15_t4_b16: QtipRaceK3L15T4Pf0Sg2B16MetalKernel,
    k2_l15_sg2_b32: QtipRaceK2L15Pf2Sg2B32MetalKernel,
    k2_l15_pf0sg4_b32: QtipRaceK2L15Pf0Sg4B32MetalKernel,
    k2_l15_r2_b32: QtipRaceK2L15R2Pf2Sg2B32MetalKernel,
    k2_l15_sg2_b64: QtipRaceK2L15Pf2Sg2B64MetalKernel,
    k2_l15_t2_b16: QtipRaceK2L15T2Pf0Sg2B16MetalKernel,
    k2_l15_t2pf2_b16: QtipRaceK2L15T2Pf2Sg2B16MetalKernel,
    s12_pf2_sg2_b32: QtipRaceV4Sign12Pf2Sg2B32MetalKernel,
    s12_pf2_sg4_b32: QtipRaceV4Sign12Pf2Sg4B32MetalKernel,
    s12_pf2_sg2_b64: QtipRaceV4Sign12Pf2Sg2B64MetalKernel,
    s12_pf2_sg4_b64: QtipRaceV4Sign12Pf2Sg4B64MetalKernel,
    s12_t2_pf2_b16: QtipRaceV4Sign12T2Pf2Sg2B16MetalKernel,
    permute_halves: QtipRacePermuteHalvesMetalKernel,
    cs_v4_b32: (QtipRaceV4CsPf2Sg4B32Pass0MetalKernel, QtipRaceV4CsPf2Sg4B32Pass1MetalKernel),
    cs_v4_r2_b32: (QtipRaceV4CsR2Pf2Sg2B32Pass0MetalKernel, QtipRaceV4CsR2Pf2Sg2B32Pass1MetalKernel),
    cs_v4_b64: (QtipRaceV4CsPf2Sg2B64Pass0MetalKernel, QtipRaceV4CsPf2Sg2B64Pass1MetalKernel),
    cs_v4_t2_b16: (QtipRaceV4CsT2Pf0Sg2B16Pass0MetalKernel, QtipRaceV4CsT2Pf0Sg2B16Pass1MetalKernel),
    // staged-activation kernels for the widest V2 shapes at batch 64
    k3_as_sg16_b64: QtipRaceK3AsPf1Sg16B64MetalKernel,
    k2_as_sg8_b64: QtipRaceK2AsPf1Sg8B64MetalKernel,
    // transposed 16-token kernels (tokens as MXU M, paired weight rows as N)
    v4_t2_pf2_b16: QtipRaceV4T2Pf2Sg2B16MetalKernel,
    k3_t4_pf0_b16: QtipRaceK3T4Pf0Sg2B16MetalKernel,
    k3_t2_pf0_b16: QtipRaceK3T2Pf0Sg2B16MetalKernel,
    k2_t2_pf0_b16: QtipRaceK2T2Pf0Sg2B16MetalKernel,
    k2_t2_pf2_b16: QtipRaceK2T2Pf2Sg2B16MetalKernel,
    transpose_bf16: QtipRowsBatchToBatchRowsBf16MetalKernel,
    transpose_f32: QtipRowsBatchToBatchRowsF32MetalKernel,
    d4_embedding: QtipD4S4EmbeddingLookupMetalKernel,
    rht32_padded: QtipRht32Bf16PaddedMetalKernel,
    i3_b16: QtipI3S4ReadoutMxuB16MetalKernel,
    i3_b32: QtipI3S4ReadoutMxuB32MetalKernel,
    i3_b64: QtipI3S4ReadoutMxuB64MetalKernel,
    i3_sparse_bf16: QtipI3S4ReadoutSparseBf16MetalKernel,
    i3_sparse_f32: QtipI3S4ReadoutSparseF32MetalKernel,
    residual_merge_hot: QtipResidualMergeHotMetalKernel,
}

impl QtipSExactMetalKernel {
    fn padded_batch(batch: u32) -> u32 {
        assert!((1..=64).contains(&batch));
        if batch <= 16 {
            16
        } else if batch <= 32 {
            32
        } else {
            64
        }
    }
}

fn env_flag(
    name: &str,
    default: bool,
) -> bool {
    std::env::var(name).map_or(default, |value| value != "0")
}

/// Race projection kernel choice (CPU-oracle-validated profile, 2026-09-01).
#[derive(Clone, Copy, Debug)]
enum RaceChoice {
    V4Pf2Sg2B32,
    V4Pf2Sg4B32,
    V4R2Pf2Sg2B32,
    V4R2Pf2Sg4B32,
    V4Pf2Sg2B64,
    V4Pf2Sg4B64,
    V4Pf2Sg8B64,
    V4PhysicalB64,
    K3Pf0Sg4B32,
    K3Pf2Sg2B32,
    K3Pf2Sg4B32,
    K3R2Pf0Sg2B32,
    K3R2Pf2Sg2B32,
    K3Pf2Sg2B64,
    K3Pf2Sg4B64,
    K2Pf0Sg4B32,
    K2Pf2Sg2B32,
    K2Pf2Sg4B32,
    K2R2Pf0Sg2B32,
    K2R2Pf2Sg2B32,
    K2Pf2Sg2B64,
    V4T2Pf2B16,
    V4L15Pf2Sg4B32,
    V4L15Pf2Sg2B32,
    V4L15R2Pf2Sg2B32,
    V4L15Pf2Sg2B64,
    V4L15Pf2Sg4B64,
    V4L15Pf2Sg8B64,
    V4L15T2Pf0B16,
    V4L15T2Pf2B16,
    V4L15T4Pf1B16,
    V4L15AsSg16B64,
    K3L15Pf2Sg2B32,
    K3L15Pf0Sg4B32,
    K3L15R2Pf2Sg2B32,
    K3L15Pf2Sg2B64,
    K3L15Pf2Sg4B64,
    K3L15T2Pf0B16,
    K3L15T4Pf0B16,
    K2L15Pf2Sg2B32,
    K2L15Pf0Sg4B32,
    K2L15R2Pf2Sg2B32,
    K2L15Pf2Sg2B64,
    K2L15T2Pf0B16,
    K2L15T2Pf2B16,
    V4S12Pf2Sg2B32,
    V4S12Pf2Sg4B32,
    V4S12Pf2Sg2B64,
    V4S12Pf2Sg4B64,
    V4S12T2Pf2B16,
    V4AntiPf2Sg2B32,
    V4AntiPf2Sg4B32,
    V4AntiPf2Sg2B64,
    V4AntiPf2Sg4B64,
    V4AntiT2Pf2B16,
    V4AntiAsSg16B64,
    V4S14Pf2Sg2B32,
    V4S14Pf2Sg4B32,
    V4S14Pf2Sg2B64,
    V4S14Pf2Sg4B64,
    V4S14T2Pf2B16,
    V4S14AsSg16B64,
    V4CsB32,
    V4CsR2B32,
    V4CsB64,
    V4CsT2B16,
    K3AsSg16B64,
    K2AsSg8B64,
    K3T4Pf0B16,
    K3T2Pf0B16,
    K2T2Pf0B16,
    K2T2Pf2B16,
}

fn select_race_kernel(
    vector_width: u32,
    transition_bits: u32,
    state_bits: u32,
    table_mode: u32,
    padded_batch: u32,
    batch: u32,
    rows: u32,
    columns: u32,
) -> RaceChoice {
    use RaceChoice::*;
    let small = batch <= 16;
    match (vector_width, transition_bits, padded_batch) {
        // V4 four-sign 16 KiB table (table_mode 3)
        (4, _, 32) if table_mode == 3 => {
            // same-run sweep 2026-09-02: batch 16 T2 527 / Sg4 536 on mlp_up, Sg4B32 best elsewhere (in_proj 527 vs T2 751)
            if small {
                if rows == 34816 { V4S12T2Pf2B16 } else { V4S12Pf2Sg4B32 }
            } else if rows == 34816 {
                V4S12Pf2Sg4B32
            } else {
                V4S12Pf2Sg2B32
            }
        },
        (4, _, 64) if table_mode == 3 => {
            if columns == 17408 {
                V4S12Pf2Sg4B64
            } else {
                V4S12Pf2Sg2B64
            }
        },
        // V2 at L=15 (32768-row table): mask-only decode
        (2, 6, 32) if state_bits == 15 => {
            if small {
                if rows == 34816 { K3L15T4Pf0B16 } else { K3L15T2Pf0B16 }
            } else if rows == 34816 {
                K3L15R2Pf2Sg2B32
            } else if rows == 16480 {
                K3L15Pf0Sg4B32
            } else {
                K3L15Pf2Sg2B32
            }
        },
        (2, 6, 64) if state_bits == 15 => K3L15Pf2Sg2B64,
        (2, 4, 32) if state_bits == 15 => {
            if small {
                K2L15T2Pf0B16
            } else if rows == 34816 {
                K2L15R2Pf2Sg2B32
            } else {
                K2L15Pf0Sg4B32
            }
        },
        (2, 4, 64) if state_bits == 15 => K2L15Pf2Sg2B64,
        // two-sign 64 KiB V4 table (table_mode 2)
        // two-sign packages: 64 KiB kernels where they win (in_proj, mlp_up at batch 64, small batch),
        // antipodal kernels on the 128 KiB half-expanded table elsewhere (same-run timings 2026-09-02)
        (4, _, 32) if table_mode == 2 => {
            if small && rows == 34816 {
                V4S14T2Pf2B16
            } else if small {
                V4S14Pf2Sg2B32
            } else if rows == 16480 {
                V4S14Pf2Sg4B32
            } else if rows == 34816 {
                V4AntiPf2Sg4B32
            } else {
                V4AntiPf2Sg2B32
            }
        },
        (4, _, 64) if table_mode == 2 => {
            if rows == 16480 || rows == 34816 {
                V4S14Pf2Sg2B64
            } else if columns == 17408 {
                V4AntiPf2Sg4B64
            } else {
                V4AntiPf2Sg2B64
            }
        },
        // antipodal L=16 V4 table (128 KiB stored half + sign select): the L=15 picks with the anti kernels
        (4, _, 32) if table_mode == 1 => {
            if small && (rows == 16480 || rows == 34816) {
                V4AntiT2Pf2B16
            } else if small || rows == 6144 {
                V4AntiPf2Sg4B32
            } else {
                V4AntiPf2Sg2B32
            }
        },
        (4, _, 64) if table_mode == 1 => {
            if columns == 17408 {
                V4AntiPf2Sg4B64
            } else if rows == 34816 {
                V4AntiAsSg16B64
            } else {
                V4AntiPf2Sg2B64
            }
        },
        // L=15 V4 leaves: one pass over a 128 KiB table, so the plain device-tensor kernels win
        // measured 2026-09-02 (min of 7, oracle-checked): suffix <= 16 pads to 32
        (4, _, 32) if state_bits == 15 => {
            if small && (rows == 16480 || rows == 34816) {
                // in_proj 177 us, mlp_up 497 us as transposed 16-token kernels
                V4L15T2Pf2B16
            } else if small || rows == 6144 {
                // gate 101/104 us, mlp_down 342 us, out_proj 99 us at suffix <= 16
                V4L15Pf2Sg4B32
            } else if rows == 34816 {
                // mlp_up 511 us
                V4L15R2Pf2Sg2B32
            } else {
                // in_proj 239 us, mlp_down 348 us, out_proj 100 us
                V4L15Pf2Sg2B32
            }
        },
        (4, _, 64) if state_bits == 15 => {
            if columns == 17408 {
                // mlp_down 405 us
                V4L15Pf2Sg4B64
            } else if rows == 34816 {
                // mlp_up 715 us with staged activations and 16 SIMDgroups (vs 865 single pass)
                V4L15AsSg16B64
            } else {
                // mlp_up 873, in_proj 393, gate 129, out_proj 123 us
                V4L15Pf2Sg2B64
            }
        },
        (4, _, 32) => {
            if columns == 17408 {
                // mlp_down 5120x17408: the component split re-streams the codes and does not pay here
                V4Pf2Sg4B32
            } else if small && rows == 16480 {
                // in_proj at suffix <= 16
                V4CsT2B16
            } else if columns == 6144 || rows == 34816 || small {
                // out_proj 5120x6144, mlp_up 34816x5120, small-suffix gate/mlp_up
                V4CsB32
            } else {
                // in_proj 16480x5120, gate 6144x5120 at suffix 17..32
                V4CsR2B32
            }
        },
        (4, _, 64) => {
            if columns == 17408 && batch == 64 {
                V4PhysicalB64
            } else if columns == 17408 {
                V4Pf2Sg4B64
            } else {
                // mlp_up, in_proj, gate, out_proj at suffix 33..64
                V4CsB64
            }
        },
        (2, 6, 32) => {
            if small {
                if rows == 34816 || rows == 16480 {
                    K3T4Pf0B16
                } else {
                    K3T2Pf0B16
                }
            } else if rows >= 16480 {
                K3R2Pf2Sg2B32
            } else if rows == 8192 {
                K3Pf2Sg2B32
            } else {
                K3Pf0Sg4B32
            }
        },
        (2, 6, 64) => {
            if rows == 34816 {
                K3AsSg16B64
            } else if rows == 16480 {
                K3Pf2Sg4B64
            } else {
                K3Pf2Sg2B64
            }
        },
        (2, 4, 32) => {
            if small {
                if columns == 17408 {
                    K2Pf2Sg2B32
                } else if rows == 34816 {
                    K2T2Pf2B16
                } else {
                    K2T2Pf0B16
                }
            } else if rows == 34816 {
                K2R2Pf2Sg2B32
            } else {
                K2Pf0Sg4B32
            }
        },
        (2, 4, 64) => {
            if rows == 34816 {
                K2AsSg8B64
            } else {
                K2Pf2Sg2B64
            }
        },
        _ => unreachable!("unsupported QTIP race configuration {vector_width}/{transition_bits}/{padded_batch}"),
    }
}

impl QtipSExactKernel<Metal> for QtipSExactMetalKernel {
    fn new(context: &MetalContext) -> Result<Self, MetalError> {
        Ok(Self {
            race_transform: env_flag("QTIP_RACE_TRANSFORM", true),
            race_projection: env_flag("QTIP_RACE_PROJ", true),
            null_projection: env_flag("QTIP_RACE_NULL_PROJ", false),
            null_transform: env_flag("QTIP_RACE_NULL_TRANSFORM", false),
            full_incoherence_a8: QtipFullIncoherenceA8MetalKernel::new(context)?,
            transform_5120: QtipRaceTransform5120MetalKernel::new(context)?,
            transform_6144: QtipRaceTransform6144MetalKernel::new(context)?,
            transform_17408: QtipRaceTransform17408MetalKernel::new(context)?,
            v2_k2_b16: QtipGaussianPhysicalQ8V2A8DirectK2B16MetalKernel::new(context)?,
            v2_k2_b32: QtipGaussianPhysicalQ8V2A8DirectK2B32MetalKernel::new(context)?,
            v2_k2_b32_batch_rows: QtipGaussianPhysicalQ8V2A8DirectK2B32BatchRowsMetalKernel::new(context)?,
            v2_k2_b64: QtipGaussianPhysicalQ8V2A8DirectK2B64MetalKernel::new(context)?,
            v2_k2_b64_batch_rows: QtipGaussianPhysicalQ8V2A8DirectK2B64BatchRowsMetalKernel::new(context)?,
            v2_k3_b16: QtipGaussianPhysicalQ8V2A8DirectK3B16MetalKernel::new(context)?,
            v2_k3_b32: QtipGaussianPhysicalQ8V2A8DirectK3B32MetalKernel::new(context)?,
            v2_k3_b32_batch_rows: QtipGaussianPhysicalQ8V2A8DirectK3B32BatchRowsMetalKernel::new(context)?,
            v2_k3_b64: QtipGaussianPhysicalQ8V2A8DirectK3B64MetalKernel::new(context)?,
            v2_k3_b64_batch_rows: QtipGaussianPhysicalQ8V2A8DirectK3B64BatchRowsMetalKernel::new(context)?,
            v4_b16: QtipGaussianPhysicalQ8V4A8DirectB16MetalKernel::new(context)?,
            v4_b32: QtipGaussianPhysicalQ8V4A8DirectB32MetalKernel::new(context)?,
            v4_b32_batch_rows: QtipGaussianPhysicalQ8V4A8DirectB32BatchRowsMetalKernel::new(context)?,
            v4_b64: QtipGaussianPhysicalQ8V4A8DirectB64MetalKernel::new(context)?,
            v4_b64_batch_rows: QtipGaussianPhysicalQ8V4A8DirectB64BatchRowsMetalKernel::new(context)?,
            v4_pf2_sg2_b32: QtipRaceV4Pf2Sg2B32MetalKernel::new(context)?,
            v4_pf2_sg4_b32: QtipRaceV4Pf2Sg4B32MetalKernel::new(context)?,
            v4_r2_pf2_sg2_b32: QtipRaceV4R2Pf2Sg2B32MetalKernel::new(context)?,
            v4_r2_pf2_sg4_b32: QtipRaceV4R2Pf2Sg4B32MetalKernel::new(context)?,
            v4_pf2_sg2_b64: QtipRaceV4Pf2Sg2B64MetalKernel::new(context)?,
            v4_pf2_sg4_b64: QtipRaceV4Pf2Sg4B64MetalKernel::new(context)?,
            v4_pf2_sg8_b64: QtipRaceV4Pf2Sg8B64MetalKernel::new(context)?,
            k3_pf0_sg4_b32: QtipRaceK3Pf0Sg4B32MetalKernel::new(context)?,
            k3_pf2_sg2_b32: QtipRaceK3Pf2Sg2B32MetalKernel::new(context)?,
            k3_pf2_sg4_b32: QtipRaceK3Pf2Sg4B32MetalKernel::new(context)?,
            k3_r2_pf0_sg2_b32: QtipRaceK3R2Pf0Sg2B32MetalKernel::new(context)?,
            k3_r2_pf2_sg2_b32: QtipRaceK3R2Pf2Sg2B32MetalKernel::new(context)?,
            k3_pf2_sg2_b64: QtipRaceK3Pf2Sg2B64MetalKernel::new(context)?,
            k3_pf2_sg4_b64: QtipRaceK3Pf2Sg4B64MetalKernel::new(context)?,
            k2_pf0_sg4_b32: QtipRaceK2Pf0Sg4B32MetalKernel::new(context)?,
            k2_pf2_sg2_b32: QtipRaceK2Pf2Sg2B32MetalKernel::new(context)?,
            k2_pf2_sg4_b32: QtipRaceK2Pf2Sg4B32MetalKernel::new(context)?,
            k2_r2_pf0_sg2_b32: QtipRaceK2R2Pf0Sg2B32MetalKernel::new(context)?,
            k2_r2_pf2_sg2_b32: QtipRaceK2R2Pf2Sg2B32MetalKernel::new(context)?,
            k2_pf2_sg2_b64: QtipRaceK2Pf2Sg2B64MetalKernel::new(context)?,
            l15_pf2_sg4_b32: QtipRaceV4L15Pf2Sg4B32MetalKernel::new(context)?,
            l15_pf2_sg2_b32: QtipRaceV4L15Pf2Sg2B32MetalKernel::new(context)?,
            l15_r2_pf2_sg2_b32: QtipRaceV4L15R2Pf2Sg2B32MetalKernel::new(context)?,
            l15_pf2_sg2_b64: QtipRaceV4L15Pf2Sg2B64MetalKernel::new(context)?,
            l15_pf2_sg4_b64: QtipRaceV4L15Pf2Sg4B64MetalKernel::new(context)?,
            l15_pf2_sg8_b64: QtipRaceV4L15Pf2Sg8B64MetalKernel::new(context)?,
            l15_t2_pf0_b16: QtipRaceV4L15T2Pf0Sg2B16MetalKernel::new(context)?,
            l15_t2_pf2_b16: QtipRaceV4L15T2Pf2Sg2B16MetalKernel::new(context)?,
            l15_t4_pf1_b16: QtipRaceV4L15T4Pf1Sg2B16MetalKernel::new(context)?,
            l15_as_sg16_b64: QtipRaceV4L15AsPf1Sg16B64MetalKernel::new(context)?,
            anti_pf2_sg2_b32: QtipRaceV4AntiPf2Sg2B32MetalKernel::new(context)?,
            anti_pf2_sg4_b32: QtipRaceV4AntiPf2Sg4B32MetalKernel::new(context)?,
            anti_pf2_sg2_b64: QtipRaceV4AntiPf2Sg2B64MetalKernel::new(context)?,
            anti_pf2_sg4_b64: QtipRaceV4AntiPf2Sg4B64MetalKernel::new(context)?,
            anti_t2_pf2_b16: QtipRaceV4AntiT2Pf2Sg2B16MetalKernel::new(context)?,
            anti_as_sg16_b64: QtipRaceV4AntiAsPf1Sg16B64MetalKernel::new(context)?,
            s14_pf2_sg2_b32: QtipRaceV4Sign14Pf2Sg2B32MetalKernel::new(context)?,
            s14_pf2_sg4_b32: QtipRaceV4Sign14Pf2Sg4B32MetalKernel::new(context)?,
            s14_pf2_sg2_b64: QtipRaceV4Sign14Pf2Sg2B64MetalKernel::new(context)?,
            s14_pf2_sg4_b64: QtipRaceV4Sign14Pf2Sg4B64MetalKernel::new(context)?,
            s14_t2_pf2_b16: QtipRaceV4Sign14T2Pf2Sg2B16MetalKernel::new(context)?,
            s14_as_sg16_b64: QtipRaceV4Sign14AsPf1Sg16B64MetalKernel::new(context)?,
            k3_l15_sg2_b32: QtipRaceK3L15Pf2Sg2B32MetalKernel::new(context)?,
            k3_l15_pf0sg4_b32: QtipRaceK3L15Pf0Sg4B32MetalKernel::new(context)?,
            k3_l15_r2_b32: QtipRaceK3L15R2Pf2Sg2B32MetalKernel::new(context)?,
            k3_l15_sg2_b64: QtipRaceK3L15Pf2Sg2B64MetalKernel::new(context)?,
            k3_l15_sg4_b64: QtipRaceK3L15Pf2Sg4B64MetalKernel::new(context)?,
            k3_l15_t2_b16: QtipRaceK3L15T2Pf0Sg2B16MetalKernel::new(context)?,
            k3_l15_t4_b16: QtipRaceK3L15T4Pf0Sg2B16MetalKernel::new(context)?,
            k2_l15_sg2_b32: QtipRaceK2L15Pf2Sg2B32MetalKernel::new(context)?,
            k2_l15_pf0sg4_b32: QtipRaceK2L15Pf0Sg4B32MetalKernel::new(context)?,
            k2_l15_r2_b32: QtipRaceK2L15R2Pf2Sg2B32MetalKernel::new(context)?,
            k2_l15_sg2_b64: QtipRaceK2L15Pf2Sg2B64MetalKernel::new(context)?,
            k2_l15_t2_b16: QtipRaceK2L15T2Pf0Sg2B16MetalKernel::new(context)?,
            k2_l15_t2pf2_b16: QtipRaceK2L15T2Pf2Sg2B16MetalKernel::new(context)?,
            s12_pf2_sg2_b32: QtipRaceV4Sign12Pf2Sg2B32MetalKernel::new(context)?,
            s12_pf2_sg4_b32: QtipRaceV4Sign12Pf2Sg4B32MetalKernel::new(context)?,
            s12_pf2_sg2_b64: QtipRaceV4Sign12Pf2Sg2B64MetalKernel::new(context)?,
            s12_pf2_sg4_b64: QtipRaceV4Sign12Pf2Sg4B64MetalKernel::new(context)?,
            s12_t2_pf2_b16: QtipRaceV4Sign12T2Pf2Sg2B16MetalKernel::new(context)?,
            permute_halves: QtipRacePermuteHalvesMetalKernel::new(context)?,
            cs_v4_b32: (
                QtipRaceV4CsPf2Sg4B32Pass0MetalKernel::new(context)?,
                QtipRaceV4CsPf2Sg4B32Pass1MetalKernel::new(context)?,
            ),
            cs_v4_r2_b32: (
                QtipRaceV4CsR2Pf2Sg2B32Pass0MetalKernel::new(context)?,
                QtipRaceV4CsR2Pf2Sg2B32Pass1MetalKernel::new(context)?,
            ),
            cs_v4_b64: (
                QtipRaceV4CsPf2Sg2B64Pass0MetalKernel::new(context)?,
                QtipRaceV4CsPf2Sg2B64Pass1MetalKernel::new(context)?,
            ),
            cs_v4_t2_b16: (
                QtipRaceV4CsT2Pf0Sg2B16Pass0MetalKernel::new(context)?,
                QtipRaceV4CsT2Pf0Sg2B16Pass1MetalKernel::new(context)?,
            ),
            k3_as_sg16_b64: QtipRaceK3AsPf1Sg16B64MetalKernel::new(context)?,
            k2_as_sg8_b64: QtipRaceK2AsPf1Sg8B64MetalKernel::new(context)?,
            v4_t2_pf2_b16: QtipRaceV4T2Pf2Sg2B16MetalKernel::new(context)?,
            k3_t4_pf0_b16: QtipRaceK3T4Pf0Sg2B16MetalKernel::new(context)?,
            k3_t2_pf0_b16: QtipRaceK3T2Pf0Sg2B16MetalKernel::new(context)?,
            k2_t2_pf0_b16: QtipRaceK2T2Pf0Sg2B16MetalKernel::new(context)?,
            k2_t2_pf2_b16: QtipRaceK2T2Pf2Sg2B16MetalKernel::new(context)?,
            transpose_bf16: QtipRowsBatchToBatchRowsBf16MetalKernel::new(context)?,
            transpose_f32: QtipRowsBatchToBatchRowsF32MetalKernel::new(context)?,
            d4_embedding: QtipD4S4EmbeddingLookupMetalKernel::new(context)?,
            rht32_padded: QtipRht32Bf16PaddedMetalKernel::new(context)?,
            i3_b16: QtipI3S4ReadoutMxuB16MetalKernel::new(context)?,
            i3_b32: QtipI3S4ReadoutMxuB32MetalKernel::new(context)?,
            i3_b64: QtipI3S4ReadoutMxuB64MetalKernel::new(context)?,
            i3_sparse_bf16: QtipI3S4ReadoutSparseBf16MetalKernel::new(context)?,
            i3_sparse_f32: QtipI3S4ReadoutSparseF32MetalKernel::new(context)?,
            residual_merge_hot: QtipResidualMergeHotMetalKernel::new(context)?,
        })
    }

    fn encode_qtip_gaussian(
        &self,
        arguments: QtipGaussianArguments<'_, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<Allocation<Metal>, MetalError> {
        let QtipGaussianArguments {
            input,
            codes,
            codebook,
            codebook_split,
            state_bits,
            table_mode,
            codebook_scale,
            scales,
            gains,
            signs,
            small_q,
            batch,
            rows,
            columns,
            vector_width,
            transition_bits,
            restart_columns,
        } = arguments;
        assert!(matches!(columns, 5120 | 6144 | 17408));
        assert_eq!(input.size(), size_for_shape(&[batch, columns], DataType::BF16));
        assert_eq!(scales.size(), size_for_shape(&[rows], DataType::F16));
        assert_eq!(gains.size(), size_for_shape(&[rows], DataType::BF16));
        assert_eq!(signs.size(), size_for_shape(&[columns], DataType::F32));
        let (power, order) = match columns {
            5120 => (1024, 5),
            6144 => (2048, 3),
            17408 => (1024, 17),
            _ => unreachable!(),
        };
        assert_eq!(small_q.size(), size_for_shape(&[order, order], DataType::F32));

        // the race projection path runs 32-token MXU tiles at minimum
        let padded_batch = if self.race_projection {
            Self::padded_batch(batch).max(32)
        } else {
            Self::padded_batch(batch)
        };
        let mut transformed_q8 = encoder.allocate_scratch(size_for_shape(&[padded_batch, columns], DataType::I8))?;
        let mut activation_scales = encoder.allocate_scratch(size_for_shape(&[padded_batch], DataType::F32))?;
        if self.null_transform {
            // probe: leave the A8 buffers uninitialised
        } else if self.race_transform {
            macro_rules! transform {
                ($kernel:expr) => {
                    $kernel.encode(
                        input,
                        signs,
                        small_q,
                        &mut transformed_q8,
                        &mut activation_scales,
                        batch,
                        padded_batch,
                        columns,
                        encoder,
                    )
                };
            }
            match columns {
                5120 => transform!(self.transform_5120),
                6144 => transform!(self.transform_6144),
                17408 => transform!(self.transform_17408),
                _ => unreachable!(),
            }
        } else {
            self.full_incoherence_a8.encode(
                input,
                signs,
                small_q,
                &mut transformed_q8,
                &mut activation_scales,
                batch,
                padded_batch,
                columns,
                order,
                power,
                encoder,
            );
        }

        let (groups, bytes_per_row) = match vector_width {
            2 => {
                assert!(matches!(transition_bits, 4 | 6));
                assert_eq!(restart_columns, 0);
                assert!(matches!(state_bits, 15 | 16));
                assert_eq!(codebook.size(), size_for_shape(&[1u32 << state_bits, 2], DataType::I8));
                let groups = columns / 2;
                (groups, (16 + (groups - 1) * transition_bits).div_ceil(8))
            },
            4 => {
                assert_eq!(transition_bits, 8);
                assert_eq!(restart_columns, 64);
                assert!(matches!(state_bits, 15 | 16));
                let stored_states = match table_mode {
                    3 => 4_096u32,
                    2 => 16_384u32,
                    1 => 32_768u32,
                    _ => 1u32 << state_bits,
                };
                assert_eq!(codebook.size(), size_for_shape(&[stored_states, 4], DataType::I8));
                (columns / 4, columns / restart_columns * 17)
            },
            _ => panic!("unsupported QTIP vector width {vector_width}"),
        };
        assert_eq!(codes.size(), rows as usize * bytes_per_row as usize);

        if self.null_projection {
            return encoder.allocate_scratch(size_for_shape(&[batch, rows], DataType::BF16));
        }
        if self.race_projection {
            let mut output = encoder.allocate_scratch(size_for_shape(&[batch, rows], DataType::BF16))?;
            let choice = select_race_kernel(vector_width, transition_bits, state_bits, table_mode, padded_batch, batch, rows, columns);
            let component_split = matches!(
                choice,
                RaceChoice::V4CsB32 | RaceChoice::V4CsR2B32 | RaceChoice::V4CsB64 | RaceChoice::V4CsT2B16
            );
            assert!(!component_split || codebook_split.is_some(), "component split requires the split V4 table");
            // int32 partials and K-packed activation halves are only used by the component split
            let mut partials = encoder.allocate_scratch(if component_split {
                size_for_shape(&[batch, rows], DataType::I32)
            } else {
                16
            })?;
            let half_size = if component_split {
                size_for_shape(&[padded_batch, columns / 2], DataType::I8)
            } else {
                16
            };
            let mut activations_lo = encoder.allocate_scratch(half_size)?;
            let mut activations_hi = encoder.allocate_scratch(half_size)?;
            if component_split {
                self.permute_halves.encode(
                    &transformed_q8,
                    &mut activations_lo,
                    &mut activations_hi,
                    padded_batch,
                    columns,
                    encoder,
                );
            }
            let split_table = codebook_split.unwrap_or(codebook);
            // antipodal kernels on a two-sign package read the half-expanded 128 KiB table
            let wide_table = if table_mode == 2 { codebook_split.unwrap_or(codebook) } else { codebook };
            macro_rules! run_wide {
                ($kernel:expr) => {
                    $kernel.encode(
                        codes,
                        wide_table,
                        &transformed_q8,
                        &activation_scales,
                        scales,
                        gains,
                        &mut output,
                        &mut partials,
                        codebook_scale,
                        rows,
                        groups,
                        bytes_per_row,
                        batch,
                        encoder,
                    )
                };
            }
            macro_rules! run_cs {
                ($pair:expr) => {{
                    $pair.0.encode(
                        codes,
                        split_table,
                        &activations_lo,
                        &activation_scales,
                        scales,
                        gains,
                        &mut output,
                        &mut partials,
                        codebook_scale,
                        rows,
                        groups,
                        bytes_per_row,
                        batch,
                        encoder,
                    );
                    $pair.1.encode(
                        codes,
                        split_table,
                        &activations_hi,
                        &activation_scales,
                        scales,
                        gains,
                        &mut output,
                        &mut partials,
                        codebook_scale,
                        rows,
                        groups,
                        bytes_per_row,
                        batch,
                        encoder,
                    );
                }};
            }
            macro_rules! run {
                ($kernel:expr) => {
                    $kernel.encode(
                        codes,
                        codebook,
                        &transformed_q8,
                        &activation_scales,
                        scales,
                        gains,
                        &mut output,
                        &mut partials,
                        codebook_scale,
                        rows,
                        groups,
                        bytes_per_row,
                        batch,
                        encoder,
                    )
                };
            }
            match choice {
                RaceChoice::V4L15Pf2Sg4B32 => run!(self.l15_pf2_sg4_b32),
                RaceChoice::V4L15Pf2Sg2B32 => run!(self.l15_pf2_sg2_b32),
                RaceChoice::V4L15R2Pf2Sg2B32 => run!(self.l15_r2_pf2_sg2_b32),
                RaceChoice::V4L15Pf2Sg2B64 => run!(self.l15_pf2_sg2_b64),
                RaceChoice::V4L15Pf2Sg4B64 => run!(self.l15_pf2_sg4_b64),
                RaceChoice::V4L15Pf2Sg8B64 => run!(self.l15_pf2_sg8_b64),
                RaceChoice::V4L15T2Pf0B16 => run!(self.l15_t2_pf0_b16),
                RaceChoice::V4L15T2Pf2B16 => run!(self.l15_t2_pf2_b16),
                RaceChoice::V4L15T4Pf1B16 => run!(self.l15_t4_pf1_b16),
                RaceChoice::V4L15AsSg16B64 => run!(self.l15_as_sg16_b64),
                RaceChoice::K3L15Pf2Sg2B32 => run!(self.k3_l15_sg2_b32),
                RaceChoice::K3L15Pf0Sg4B32 => run!(self.k3_l15_pf0sg4_b32),
                RaceChoice::K3L15R2Pf2Sg2B32 => run!(self.k3_l15_r2_b32),
                RaceChoice::K3L15Pf2Sg2B64 => run!(self.k3_l15_sg2_b64),
                RaceChoice::K3L15Pf2Sg4B64 => run!(self.k3_l15_sg4_b64),
                RaceChoice::K3L15T2Pf0B16 => run!(self.k3_l15_t2_b16),
                RaceChoice::K3L15T4Pf0B16 => run!(self.k3_l15_t4_b16),
                RaceChoice::K2L15Pf2Sg2B32 => run!(self.k2_l15_sg2_b32),
                RaceChoice::K2L15Pf0Sg4B32 => run!(self.k2_l15_pf0sg4_b32),
                RaceChoice::K2L15R2Pf2Sg2B32 => run!(self.k2_l15_r2_b32),
                RaceChoice::K2L15Pf2Sg2B64 => run!(self.k2_l15_sg2_b64),
                RaceChoice::K2L15T2Pf0B16 => run!(self.k2_l15_t2_b16),
                RaceChoice::K2L15T2Pf2B16 => run!(self.k2_l15_t2pf2_b16),
                RaceChoice::V4S12Pf2Sg2B32 => run!(self.s12_pf2_sg2_b32),
                RaceChoice::V4S12Pf2Sg4B32 => run!(self.s12_pf2_sg4_b32),
                RaceChoice::V4S12Pf2Sg2B64 => run!(self.s12_pf2_sg2_b64),
                RaceChoice::V4S12Pf2Sg4B64 => run!(self.s12_pf2_sg4_b64),
                RaceChoice::V4S12T2Pf2B16 => run!(self.s12_t2_pf2_b16),
                RaceChoice::V4AntiPf2Sg2B32 => run_wide!(self.anti_pf2_sg2_b32),
                RaceChoice::V4AntiPf2Sg4B32 => run_wide!(self.anti_pf2_sg4_b32),
                RaceChoice::V4AntiPf2Sg2B64 => run_wide!(self.anti_pf2_sg2_b64),
                RaceChoice::V4AntiPf2Sg4B64 => run_wide!(self.anti_pf2_sg4_b64),
                RaceChoice::V4AntiT2Pf2B16 => run_wide!(self.anti_t2_pf2_b16),
                RaceChoice::V4AntiAsSg16B64 => run_wide!(self.anti_as_sg16_b64),
                RaceChoice::V4S14Pf2Sg2B32 => run!(self.s14_pf2_sg2_b32),
                RaceChoice::V4S14Pf2Sg4B32 => run!(self.s14_pf2_sg4_b32),
                RaceChoice::V4S14Pf2Sg2B64 => run!(self.s14_pf2_sg2_b64),
                RaceChoice::V4S14Pf2Sg4B64 => run!(self.s14_pf2_sg4_b64),
                RaceChoice::V4S14T2Pf2B16 => run!(self.s14_t2_pf2_b16),
                RaceChoice::V4S14AsSg16B64 => run!(self.s14_as_sg16_b64),
                RaceChoice::V4CsB32 => run_cs!(self.cs_v4_b32),
                RaceChoice::V4CsR2B32 => run_cs!(self.cs_v4_r2_b32),
                RaceChoice::V4CsB64 => run_cs!(self.cs_v4_b64),
                RaceChoice::V4CsT2B16 => run_cs!(self.cs_v4_t2_b16),
                RaceChoice::K3AsSg16B64 => run!(self.k3_as_sg16_b64),
                RaceChoice::K2AsSg8B64 => run!(self.k2_as_sg8_b64),
                RaceChoice::V4Pf2Sg2B32 => run!(self.v4_pf2_sg2_b32),
                RaceChoice::V4Pf2Sg4B32 => run!(self.v4_pf2_sg4_b32),
                RaceChoice::V4R2Pf2Sg2B32 => run!(self.v4_r2_pf2_sg2_b32),
                RaceChoice::V4R2Pf2Sg4B32 => run!(self.v4_r2_pf2_sg4_b32),
                RaceChoice::V4Pf2Sg2B64 => run!(self.v4_pf2_sg2_b64),
                RaceChoice::V4Pf2Sg4B64 => run!(self.v4_pf2_sg4_b64),
                RaceChoice::V4Pf2Sg8B64 => run!(self.v4_pf2_sg8_b64),
                RaceChoice::V4PhysicalB64 => {
                    // identical batch-rows layout for batch == padded == 64
                    self.v4_b64_batch_rows.encode(
                        codes,
                        codebook,
                        &transformed_q8,
                        &activation_scales,
                        scales,
                        gains,
                        &mut output,
                        codebook_scale,
                        rows,
                        groups,
                        bytes_per_row,
                        encoder,
                    )
                },
                RaceChoice::K3Pf0Sg4B32 => run!(self.k3_pf0_sg4_b32),
                RaceChoice::K3Pf2Sg2B32 => run!(self.k3_pf2_sg2_b32),
                RaceChoice::K3Pf2Sg4B32 => run!(self.k3_pf2_sg4_b32),
                RaceChoice::K3R2Pf0Sg2B32 => run!(self.k3_r2_pf0_sg2_b32),
                RaceChoice::K3R2Pf2Sg2B32 => run!(self.k3_r2_pf2_sg2_b32),
                RaceChoice::K3Pf2Sg2B64 => run!(self.k3_pf2_sg2_b64),
                RaceChoice::K3Pf2Sg4B64 => run!(self.k3_pf2_sg4_b64),
                RaceChoice::K2Pf0Sg4B32 => run!(self.k2_pf0_sg4_b32),
                RaceChoice::K2Pf2Sg2B32 => run!(self.k2_pf2_sg2_b32),
                RaceChoice::K2Pf2Sg4B32 => run!(self.k2_pf2_sg4_b32),
                RaceChoice::K2R2Pf0Sg2B32 => run!(self.k2_r2_pf0_sg2_b32),
                RaceChoice::K2R2Pf2Sg2B32 => run!(self.k2_r2_pf2_sg2_b32),
                RaceChoice::K2Pf2Sg2B64 => run!(self.k2_pf2_sg2_b64),
                RaceChoice::V4T2Pf2B16 => run!(self.v4_t2_pf2_b16),
                RaceChoice::K3T4Pf0B16 => run!(self.k3_t4_pf0_b16),
                RaceChoice::K3T2Pf0B16 => run!(self.k3_t2_pf0_b16),
                RaceChoice::K2T2Pf0B16 => run!(self.k2_t2_pf0_b16),
                RaceChoice::K2T2Pf2B16 => run!(self.k2_t2_pf2_b16),
            }
            return Ok(output);
        }

        let direct_batch_rows = batch == padded_batch && padded_batch >= 32;
        let mut rows_batch = encoder.allocate_scratch(size_for_shape(&[rows, padded_batch], DataType::BF16))?;
        macro_rules! run {
            ($kernel:expr) => {
                $kernel.encode(
                    codes,
                    codebook,
                    &transformed_q8,
                    &activation_scales,
                    scales,
                    gains,
                    &mut rows_batch,
                    codebook_scale,
                    rows,
                    groups,
                    bytes_per_row,
                    encoder,
                )
            };
        }
        match (vector_width, transition_bits, padded_batch, direct_batch_rows) {
            (2, 4, 16, false) => run!(self.v2_k2_b16),
            (2, 4, 32, false) => run!(self.v2_k2_b32),
            (2, 4, 32, true) => run!(self.v2_k2_b32_batch_rows),
            (2, 4, 64, false) => run!(self.v2_k2_b64),
            (2, 4, 64, true) => run!(self.v2_k2_b64_batch_rows),
            (2, 6, 16, false) => run!(self.v2_k3_b16),
            (2, 6, 32, false) => run!(self.v2_k3_b32),
            (2, 6, 32, true) => run!(self.v2_k3_b32_batch_rows),
            (2, 6, 64, false) => run!(self.v2_k3_b64),
            (2, 6, 64, true) => run!(self.v2_k3_b64_batch_rows),
            (4, _, 16, false) => run!(self.v4_b16),
            (4, _, 32, false) => run!(self.v4_b32),
            (4, _, 32, true) => run!(self.v4_b32_batch_rows),
            (4, _, 64, false) => run!(self.v4_b64),
            (4, _, 64, true) => run!(self.v4_b64_batch_rows),
            _ => unreachable!(),
        }

        if direct_batch_rows {
            return Ok(rows_batch);
        }
        let mut output = encoder.allocate_scratch(size_for_shape(&[batch, rows], DataType::BF16))?;
        self.transpose_bf16.encode(&rows_batch, &mut output, batch, padded_batch, rows, encoder);
        Ok(output)
    }

    fn encode_d4_s4_embedding(
        &self,
        arguments: D4S4EmbeddingArguments<'_, Metal>,
        encoder: &mut Encoder<Metal>,
    ) {
        self.d4_embedding.encode(
            arguments.token_ids,
            arguments.codes,
            arguments.row_scales,
            arguments.ladder_indices,
            arguments.table,
            arguments.ladder,
            arguments.output_hadamard_factors,
            arguments.output,
            arguments.batch,
            arguments.vocab_size,
            arguments.model_dim,
            arguments.input_scale,
            encoder,
        );
    }

    fn encode_i3_s4_readout(
        &self,
        arguments: I3S4ReadoutArguments<'_, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<Allocation<Metal>, MetalError> {
        let I3S4ReadoutArguments {
            input,
            codes,
            row_scales,
            ladder_indices,
            ladder,
            input_hadamard_factors,
            batch,
            vocab_size,
            model_dim,
            output_data_type,
        } = arguments;
        assert_eq!(input.size(), size_for_shape(&[batch, model_dim], DataType::BF16));
        assert_eq!(codes.size(), vocab_size as usize * (model_dim as usize * 3 / 8));
        assert_eq!(row_scales.size(), size_for_shape(&[vocab_size], DataType::BF16));
        assert_eq!(ladder_indices.size(), vocab_size as usize * (model_dim as usize / 128));
        assert_eq!(ladder.size(), size_for_shape(&[16], DataType::F16));
        assert_eq!(input_hadamard_factors.size(), size_for_shape(&[model_dim], DataType::I32));

        let padded_batch = Self::padded_batch(batch);
        let mut transformed = encoder.allocate_scratch(size_for_shape(&[padded_batch, model_dim], DataType::BF16))?;
        self.rht32_padded.encode(
            input,
            input_hadamard_factors,
            &mut transformed,
            batch,
            padded_batch,
            model_dim,
            encoder,
        );
        let mut rows_batch = encoder.allocate_scratch(size_for_shape(&[vocab_size, padded_batch], DataType::BF16))?;
        let code_stride = model_dim * 3 / 8;
        let ladder_stride = model_dim / 128;
        macro_rules! run {
            ($kernel:expr) => {
                $kernel.encode(
                    codes,
                    &transformed,
                    row_scales,
                    ladder_indices,
                    ladder,
                    &mut rows_batch,
                    vocab_size,
                    model_dim,
                    code_stride,
                    ladder_stride,
                    encoder,
                )
            };
        }
        match padded_batch {
            16 => run!(self.i3_b16),
            32 => run!(self.i3_b32),
            64 => run!(self.i3_b64),
            _ => unreachable!(),
        }

        let mut output = encoder.allocate_scratch(size_for_shape(&[batch, vocab_size], output_data_type))?;
        match output_data_type {
            DataType::BF16 => {
                self.transpose_bf16.encode(&rows_batch, &mut output, batch, padded_batch, vocab_size, encoder)
            },
            DataType::F32 => {
                self.transpose_f32.encode(&rows_batch, &mut output, batch, padded_batch, vocab_size, encoder)
            },
            _ => panic!("unsupported i3 readout output type {output_data_type:?}"),
        }
        Ok(output)
    }

    fn encode_i3_s4_readout_sparse(
        &self,
        arguments: I3S4SparseReadoutArguments<'_, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<Allocation<Metal>, MetalError> {
        let I3S4SparseReadoutArguments {
            input,
            token_ids,
            codes,
            row_scales,
            ladder_indices,
            ladder,
            input_hadamard_factors,
            rows,
            ids_per_row,
            vocab_size,
            model_dim,
            output_data_type,
            soft_cap,
        } = arguments;
        assert!(rows > 0 && ids_per_row > 0);
        assert!(model_dim % 256 == 0 && model_dim <= 8192, "sparse i3 readout expects model_dim % 256 == 0");
        // the weaver hands over buffers sized for its maximum node count; only the first `rows` rows are read
        assert!(input.size() >= size_for_shape(&[rows, model_dim], DataType::BF16));
        assert!(token_ids.size() >= size_for_shape(&[rows, ids_per_row], DataType::U32));
        assert_eq!(codes.size(), vocab_size as usize * (model_dim as usize * 3 / 8));
        assert_eq!(row_scales.size(), size_for_shape(&[vocab_size], DataType::BF16));
        assert_eq!(ladder_indices.size(), vocab_size as usize * (model_dim as usize / 128));
        assert_eq!(ladder.size(), size_for_shape(&[16], DataType::F16));
        assert_eq!(input_hadamard_factors.size(), size_for_shape(&[model_dim], DataType::I32));

        // rht32 pads to whole 16-row blocks; the sparse kernel only reads the first `rows` rows
        let padded_rows = rows.div_ceil(16) * 16;
        let mut transformed = encoder.allocate_scratch(size_for_shape(&[padded_rows, model_dim], DataType::BF16))?;
        self.rht32_padded.encode(input, input_hadamard_factors, &mut transformed, rows, padded_rows, model_dim, encoder);

        let mut output = encoder.allocate_scratch(size_for_shape(&[rows, ids_per_row], output_data_type))?;
        let code_stride = model_dim * 3 / 8;
        let ladder_stride = model_dim / 128;
        macro_rules! run {
            ($kernel:expr) => {
                $kernel.encode(
                    codes,
                    &transformed,
                    row_scales,
                    ladder_indices,
                    ladder,
                    token_ids,
                    &mut output,
                    rows,
                    ids_per_row,
                    model_dim,
                    code_stride,
                    ladder_stride,
                    soft_cap,
                    encoder,
                )
            };
        }
        match output_data_type {
            DataType::BF16 => run!(self.i3_sparse_bf16),
            DataType::F32 => run!(self.i3_sparse_f32),
            _ => panic!("unsupported sparse i3 readout output type {output_data_type:?}"),
        }
        Ok(output)
    }

    fn encode_residual_merge_hot(
        &self,
        hot: &Allocation<Metal>,
        cold: &Allocation<Metal>,
        token_ids: &Allocation<Metal>,
        output: &mut Allocation<Metal>,
        hot_rows: u32,
        count: u32,
        encoder: &mut Encoder<Metal>,
    ) {
        assert!(hot.size() >= size_for_shape(&[count], DataType::BF16));
        assert!(cold.size() >= size_for_shape(&[count], DataType::BF16));
        assert!(token_ids.size() >= size_for_shape(&[count], DataType::U32));
        assert!(output.size() >= size_for_shape(&[count], DataType::BF16));
        self.residual_merge_hot.encode(hot, cold, token_ids, output, hot_rows, count, encoder);
    }
}
