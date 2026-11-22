//! Loading of the router and the stacked expert matrices of a mixture-of-experts block.

use half::bf16;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Context, CpuBuffer, GlobalBufferCpuGpu,
        gpu_types::moe::{MoeExpertFormat, nvfp4_repack_word_fast},
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout,
        microfloat_spec::{MicrofloatScaleMode, MicrofloatSpec},
        mlx_spec::MLXSpec,
    },
    data_type::{DataType, F8E4M3},
    safetensors::{ParameterLoaderError, ParameterTree},
};

pub const NVFP4_GROUP_SIZE: u32 = 16;
pub const AFFINE_GROUP_SIZE: u32 = 64;

#[derive(Debug, Error)]
pub enum MoeWeightsError<B: Backend> {
    #[error("Parameter loading error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Unsupported mixture-of-experts configuration: {0}")]
    Unsupported(String),
}

/// One projection of every routed expert, stacked as `[experts, rows, cols]` with `cols` the input dimension.
///
/// `codes` holds the 4-bit codes of each row, eight per little-endian `u32`. Affine codes keep the checkpoint order
/// (two per byte, low nibble first); NVFP4 words are rewritten by [`repack_nvfp4_codes`]. For
/// [`MoeExpertFormat::Nvfp4`], `scales` holds one non-negative E4M3 byte per 16 codes (`[experts, rows, cols / 16]`)
/// and `global_scales` one float32 per expert. For [`MoeExpertFormat::AffineU4`], `scales` holds one little-endian
/// `u32` per 64 codes (`[experts, rows, cols / 64]`) packing the bfloat16 scale in its low half and the bfloat16 bias
/// in its high half.
pub struct ExpertWeights<B: Backend> {
    pub format: MoeExpertFormat,
    pub codes: B::GlobalGpuBuffer,
    pub scales: B::GlobalGpuBuffer,
    pub global_scales: Option<B::GlobalGpuBuffer>,
    pub rows: u32,
    pub cols: u32,
}

impl<B: Backend> ExpertWeights<B> {
    /// Loads the stacked matrix stored under `tree` (`<projection>.weights`), whose spec sits in `tree.spec`.
    pub async fn load(
        experts: u32,
        rows: u32,
        cols: u32,
        tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, MoeWeightsError<B>> {
        let spec = tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let codes_shape = [experts, rows, cols / 2];
        match spec {
            AnyWeightMatrixSpec::MicrofloatSpec(MicrofloatSpec {
                bits: 4,
                group_size: NVFP4_GROUP_SIZE,
                scale_mode: MicrofloatScaleMode::Nvfp4,
                layout: Layout::OutputInput,
                ..
            }) => {
                check_divisible::<B>(cols, NVFP4_GROUP_SIZE)?;
                let mut scales = tree.read_slice::<F8E4M3>("scales", &[experts, rows, cols / NVFP4_GROUP_SIZE]).await?;
                let negated_groups = make_scales_non_negative(&mut scales);
                let codes = tree
                    .read_buffer_with("weights", &codes_shape, DataType::U8, |codes| {
                        negate_groups(codes, &negated_groups);
                        repack_nvfp4_codes(codes);
                    })
                    .await?;
                let scales = upload::<B, F8E4M3>(&scales, context)?;
                let global_scales = tree.read_buffer("global_scale", &[experts], DataType::F32).await?;
                Ok(Self {
                    format: MoeExpertFormat::Nvfp4,
                    codes,
                    scales,
                    global_scales: Some(global_scales),
                    rows,
                    cols,
                })
            },
            AnyWeightMatrixSpec::MLXSpec(MLXSpec {
                bits: 4,
                group_size: AFFINE_GROUP_SIZE,
                layout: Layout::OutputInput,
                ..
            }) => {
                check_divisible::<B>(cols, AFFINE_GROUP_SIZE)?;
                let codes = tree.read_buffer("weights", &codes_shape, DataType::U8).await?;
                // Output-input MLX matrices store their parameters group-major, rows padded to a multiple of 4.
                let groups = cols / AFFINE_GROUP_SIZE;
                let padded_rows = rows.next_multiple_of(4);
                let parameter_shape = [experts, groups, padded_rows];
                let scales = tree.read_slice::<bf16>("scales", &parameter_shape).await?;
                let biases = tree.read_slice::<bf16>("biases", &parameter_shape).await?;
                let packed = pack_affine_parameters(&scales, &biases, experts, rows, padded_rows, groups);
                Ok(Self {
                    format: MoeExpertFormat::AffineU4,
                    codes,
                    scales: upload::<B, u32>(&packed, context)?,
                    global_scales: None,
                    rows,
                    cols,
                })
            },
            spec => Err(MoeWeightsError::Unsupported(format!(
                "expert weights must be NVFP4 (MicrofloatSpec, 4 bits, groups of 16, output-input) or MLX affine \
                 (4 bits, groups of 64, output-input), got {spec:?}"
            ))),
        }
    }
}

/// Clears the sign of every negative E4M3 scale and returns the groups it cleared, whose codes must flip sign:
/// e2m1(c) * -s = e2m1(c ^ 8) * s exactly.
pub fn make_scales_non_negative(scales: &mut [F8E4M3]) -> Vec<usize> {
    let mut negated = Vec::new();
    for (group, scale) in scales.iter_mut().enumerate() {
        if scale.0 & 0x80 != 0 {
            scale.0 &= 0x7F;
            negated.push(group);
        }
    }
    negated
}

/// Flips the sign bit of the 16 checkpoint-order NVFP4 codes (8 bytes) of each listed group.
pub fn negate_groups(
    codes: &mut [u8],
    groups: &[usize],
) {
    for &group in groups {
        for byte in &mut codes[group * 8..group * 8 + 8] {
            *byte ^= 0x88;
        }
    }
}

/// Rewrites checkpoint-order NVFP4 codes in place into the layout the expert kernels decode (see
/// [`crate::backends::common::gpu_types::moe::nvfp4_repack_word`]). Every word is rewritten on its own.
pub fn repack_nvfp4_codes(codes: &mut [u8]) {
    assert!(codes.len().is_multiple_of(4), "NVFP4 rows hold whole words");
    let repack = |chunk: &mut [u8]| match bytemuck::try_cast_slice_mut::<u8, u32>(chunk) {
        Ok(words) => {
            for word in words {
                *word = nvfp4_repack_word_fast(u32::from_le(*word)).to_le();
            }
        },
        Err(_) => {
            for bytes in chunk.as_chunks_mut::<4>().0 {
                *bytes = nvfp4_repack_word_fast(u32::from_le_bytes(*bytes)).to_le_bytes();
            }
        },
    };
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let chunk_bytes = codes.len().div_ceil(threads).next_multiple_of(1 << 16);
    if threads == 1 || codes.len() <= chunk_bytes {
        repack(codes);
        return;
    }
    std::thread::scope(|scope| {
        for chunk in codes.chunks_mut(chunk_bytes) {
            scope.spawn(move || repack(chunk));
        }
    });
}

fn check_divisible<B: Backend>(
    cols: u32,
    group_size: u32,
) -> Result<(), MoeWeightsError<B>> {
    if cols.is_multiple_of(group_size) {
        Ok(())
    } else {
        Err(MoeWeightsError::Unsupported(format!("expert input dimension {cols} is not a multiple of {group_size}")))
    }
}

/// Repacks group-major `[experts, groups, padded_rows]` scales and biases into row-major `[experts, rows, groups]`
/// words, the scale in the low half. Experts repack in parallel, rows in cache-sized blocks.
pub fn pack_affine_parameters(
    scales: &[bf16],
    biases: &[bf16],
    experts: u32,
    rows: u32,
    padded_rows: u32,
    groups: u32,
) -> Vec<u32> {
    const BLOCK_ROWS: usize = 64;
    let (experts, rows, padded_rows, groups) = (experts as usize, rows as usize, padded_rows as usize, groups as usize);
    assert_eq!(scales.len(), experts * groups * padded_rows);
    assert_eq!(biases.len(), scales.len());
    let mut packed = vec![0u32; experts * rows * groups];
    if packed.is_empty() {
        return packed;
    }
    let pack_expert = |expert: usize, destination: &mut [u32]| {
        let source = expert * groups * padded_rows;
        for first_row in (0..rows).step_by(BLOCK_ROWS) {
            let block_rows = BLOCK_ROWS.min(rows - first_row);
            for group in 0..groups {
                let base = source + group * padded_rows + first_row;
                for row in 0..block_rows {
                    let scale = u32::from(scales[base + row].to_bits());
                    let bias = u32::from(biases[base + row].to_bits());
                    destination[(first_row + row) * groups + group] = scale | (bias << 16);
                }
            }
        }
    };
    let threads = std::thread::available_parallelism().map_or(1, usize::from).min(experts);
    let experts_per_thread = experts.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk, destination) in packed.chunks_mut(experts_per_thread * rows * groups).enumerate() {
            scope.spawn(move || {
                for (offset, expert_destination) in destination.chunks_mut(rows * groups).enumerate() {
                    pack_expert(chunk * experts_per_thread + offset, expert_destination);
                }
            });
        }
    });
    packed
}

pub fn upload<B: Backend, T: bytemuck::NoUninit + bytemuck::AnyBitPattern>(
    data: &[T],
    context: &B::Context,
) -> Result<B::GlobalGpuBuffer, MoeWeightsError<B>> {
    let mut buffer = context.create_buffer_cpu_gpu(size_of_val(data)).map_err(MoeWeightsError::Backend)?;
    buffer.copyin(data);
    buffer.handoff().map_err(MoeWeightsError::Backend)
}
