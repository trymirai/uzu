use std::sync::Arc;

use half::{bf16, f16};

use crate::{
    backends::common::{
        Allocation, Backend, Context, Encoder, Kernels,
        kernel::{
            matmul::{QuantParams, QuantParamsLayout},
            mirai_s::{MiraiSProjection, MiraiSTransform, ProjectionArguments, TrellisCodec, mixing_order},
        },
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout,
        i3_s4_spec::I3S4Spec,
        qtip_gaussian_spec::{PostGainAxis, QtipGaussianSpec},
        row_stack_spec::RowStackSpec,
    },
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearMatmul, LinearMatmulError},
        weight_matrix::WeightMatrix,
    },
    parameters::ParameterTree,
};

pub(crate) const TRELLIS_STATES: usize = 1 << 16;
const READOUT_GROUP_SIZE: u32 = 64;

struct Part<B: Backend> {
    projection: <B::Kernels as Kernels>::MiraiSProjection,
    codes: Allocation<B>,
    /// f32 `[rows]`: scales * gains * post gains.
    row_scales: Allocation<B>,
    /// `codebook_table` of the package codebook, shared by every leaf of the same vector width.
    codebook: Arc<Allocation<B>>,
    rows: u32,
}

/// Linear over Mirai S trellis weights: the input is rotated and quantized once, then every part writes
/// its rows of the output.
pub struct MiraiSLinear<B: Backend> {
    transform: <B::Kernels as Kernels>::MiraiSTransform,
    signs: Arc<Allocation<B>>,
    mixing: Arc<Allocation<B>>,
    parts: Box<[Part<B>]>,
    output_dimension: u32,
}

impl<B: Backend> MiraiSLinear<B> {
    /// Loads a `QtipGaussianSpec` leaf or a `RowStackSpec` of them (`parts.<index>`) from `tree`; the package's
    /// `qtip_shared` tensors live at the root.
    pub fn load(
        context: &B::Context,
        spec: AnyWeightMatrixSpec,
        tree: &ParameterTree<B>,
        input_dimension: u32,
        output_dimension: u32,
    ) -> Result<Self, LinearMatmulError<B>> {
        let parts = match spec {
            AnyWeightMatrixSpec::QtipGaussianSpec(spec) => vec![(output_dimension, spec, tree.clone())],
            AnyWeightMatrixSpec::RowStackSpec(RowStackSpec {
                parts,
                layout: Layout::OutputInput,
                ..
            }) => parts
                .into_iter()
                .enumerate()
                .map(|(index, (rows, spec))| (rows, spec, tree.subtree(&format!("parts.{index}"))))
                .collect(),
            spec => return Err(LinearMatmulError::UnsupportedConfiguration(format!("{spec:?}"))),
        };
        if parts.iter().map(|(rows, ..)| rows).sum::<u32>() != output_dimension {
            return Err(LinearMatmulError::UnsupportedConfiguration(format!(
                "row stack parts do not add up to {output_dimension} rows"
            )));
        }
        let transform = <B::Kernels as Kernels>::MiraiSTransform::new(context, input_dimension)
            .map_err(LinearMatmulError::BackendError)?
            .ok_or_else(|| {
                LinearMatmulError::UnsupportedConfiguration(format!(
                    "no Mirai S transform for {input_dimension} columns on this device"
                ))
            })?;
        let shared = tree.root().subtree("qtip_shared");
        let order = mixing_order(input_dimension);
        let signs = shared.shared_allocation(&format!("signs_{input_dimension}"), |leaf| {
            leaf.validate(&[input_dimension], DataType::F32)?.read_allocation()
        })?;
        let mixing = shared.shared_allocation(&format!("q_{input_dimension}"), |leaf| {
            leaf.validate(&[order, order], DataType::F32)?.read_allocation()
        })?;
        let parts = parts
            .into_iter()
            .map(|(rows, spec, tree)| load_part(context, &spec, rows, input_dimension, &tree, &shared))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            transform,
            signs,
            mixing,
            parts,
            output_dimension,
        })
    }
}

fn load_part<B: Backend>(
    context: &B::Context,
    spec: &QtipGaussianSpec,
    rows: u32,
    columns: u32,
    tree: &ParameterTree<B>,
    shared: &ParameterTree<B>,
) -> Result<Part<B>, LinearMatmulError<B>> {
    if spec.layout != Layout::OutputInput || !rows.is_multiple_of(16) {
        return Err(LinearMatmulError::UnsupportedConfiguration(format!("{rows} rows of {spec:?}")));
    }
    let codec = match (spec.vector_width, spec.transition_bits, spec.restart_columns) {
        (4, 8, 64) => TrellisCodec::Vector4Restart64,
        (2, 6, 0) => TrellisCodec::Vector2Transition6,
        (2, 4, 0) => TrellisCodec::Vector2Transition4,
        _ => return Err(LinearMatmulError::UnsupportedConfiguration(format!("{spec:?}"))),
    };
    let projection = <B::Kernels as Kernels>::MiraiSProjection::new(context, codec)
        .map_err(LinearMatmulError::BackendError)?
        .ok_or_else(|| LinearMatmulError::UnsupportedConfiguration("no Mirai S projection on this device".into()))?;
    let mut codes = tree.leaf("codes")?.validate(&[rows, codec.row_bytes(columns)], DataType::U8)?.read_allocation()?;
    if codec.vector_width() == 2 {
        repack_msb_first(codes.as_slice_mut(), codec, columns);
    }

    let scales = tree.leaf("scales")?.validate(&[rows], spec.scale_dtype)?;
    let mut row_scales: Vec<f32> = match spec.scale_dtype {
        DataType::F16 => scales.read_slice::<f16>()?.iter().map(|scale| scale.to_f32()).collect(),
        DataType::F32 => scales.read_slice::<f32>()?.into(),
        dtype => return Err(LinearMatmulError::UnsupportedDataType(dtype)),
    };
    let gains = tree.leaf("gains")?.validate(&[rows], DataType::BF16)?.read_slice::<bf16>()?;
    row_scales.iter_mut().zip(&gains).for_each(|(scale, gain)| *scale *= gain.to_f32());
    // a per-row gain after the rotation folds into the row scale
    for (index, PostGainAxis::Row) in spec.post_gain_axes.iter().enumerate() {
        let gains = tree.leaf(&format!("post_gains.{index}"))?.validate(&[rows], DataType::F32)?.read_slice::<f32>()?;
        row_scales.iter_mut().zip(&gains).for_each(|(scale, gain)| *scale *= gain);
    }

    let vector_width = codec.vector_width();
    let codebook = shared.shared_allocation(&format!("codebook_v{vector_width}"), |leaf| {
        let values = leaf.validate(&[TRELLIS_STATES as u32, vector_width], DataType::F32)?.read_slice::<f32>()?;
        let table = codebook_table(&values, vector_width as usize).map_err(|error| {
            LinearMatmulError::UnsupportedConfiguration(format!("codebook_v{vector_width}: {error}"))
        })?;
        context.create_allocation_from_slice(&table).map_err(LinearMatmulError::BackendError)
    })?;

    Ok(Part {
        projection,
        codes,
        row_scales: context.create_allocation_from_slice(&row_scales).map_err(LinearMatmulError::BackendError)?,
        codebook,
        rows,
    })
}

/// Codebook levels of the four bytes of fmix32(state * 0xCFCCB83F + 0x584B4AA3) (`trellis_levels` in
/// projection.metal).
pub(crate) fn trellis_levels(state: u32) -> [i32; 4] {
    let mut x = state.wrapping_mul(0xCFCC_B83F).wrapping_add(0x584B_4AA3);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 16;
    std::array::from_fn(|byte| {
        let byte = (x >> (8 * byte)) & 0xFF;
        let field_sum = (0..4).map(|field| (byte >> (2 * field)) & 3).sum::<u32>();
        (8 * field_sum + ((3 * (byte & 15)) & 15)) as i32 - 54
    })
}

/// The package codebook as the projection kernels read it: f32 `[scale, offset of column class 0..4, 0, 0, 0]`, then
/// for V2 the int8 level pair of every state. Errors unless every entry is scale * level + offset (the kernels hash levels).
pub(crate) fn codebook_table(
    values: &[f32],
    vector_width: usize,
) -> Result<Vec<u8>, String> {
    assert_eq!(values.len(), TRELLIS_STATES * vector_width);
    let levels: Vec<[i32; 4]> = (0..TRELLIS_STATES as u32).map(trellis_levels).collect();
    let value = |state: usize, component: usize| values[state * vector_width + component] as f64;
    // the state farthest from state 0 in component 0 pins the scale, state 0 then pins the offsets
    let far = (0..TRELLIS_STATES).max_by_key(|&state| (levels[state][0] - levels[0][0]).abs()).unwrap();
    let scale = (value(far, 0) - value(0, 0)) / (levels[far][0] - levels[0][0]) as f64;
    let offsets: Vec<f64> =
        (0..vector_width).map(|component| value(0, component) - scale * levels[0][component] as f64).collect();
    for state in 0..TRELLIS_STATES {
        for component in 0..vector_width {
            let error = value(state, component) - (scale * levels[state][component] as f64 + offsets[component]);
            if error.abs() > 1e-5 {
                return Err(format!("entry ({state}, {component}) is not a computed level (error {error})"));
            }
        }
    }
    let class_offset = |class: usize| offsets[class % vector_width] as f32;
    let header = [scale as f32, class_offset(0), class_offset(1), class_offset(2), class_offset(3), 0.0, 0.0, 0.0];
    let mut table = bytemuck::cast_slice::<f32, u8>(&header).to_vec();
    if vector_width == 2 {
        table.extend(levels.iter().flat_map(|&[first, second, ..]| [first as i8 as u8, second as i8 as u8]));
    }
    Ok(table)
}

/// Rewrites V2 rows in place from the package layout (16-bit little-endian seed, then transitions packed LSB
/// first) MSB first, so the state of group g is the 16-bit big-endian window at bit g * transition_bits.
fn repack_msb_first(
    codes: &mut [u8],
    codec: TrellisCodec,
    columns: u32,
) {
    let transitions = (columns / 2 - 1) as usize;
    for row in codes.chunks_exact_mut(codec.row_bytes(columns) as usize) {
        row.swap(0, 1);
        match codec {
            TrellisCodec::Vector2Transition4 => {
                // two transitions per byte: swap the nibbles
                for byte in &mut row[2..] {
                    *byte = byte.rotate_left(4);
                }
            },
            TrellisCodec::Vector2Transition6 => {
                // four transitions per three bytes; the last block is padded with zero bits
                let (blocks, []) = row[2..].as_chunks_mut::<3>() else {
                    panic!("V2 rows of {columns} columns are not whole 3-byte blocks");
                };
                for (block, bytes) in blocks.iter_mut().enumerate() {
                    let [b0, b1, b2] = *bytes;
                    let t0 = b0 & 0x3F;
                    let t1 = (b0 >> 6) | ((b1 & 0x0F) << 2);
                    let t2 = (b1 >> 4) | ((b2 & 0x03) << 4);
                    let t3 = if 4 * block + 3 < transitions {
                        b2 >> 2
                    } else {
                        0
                    };
                    *bytes = [(t0 << 2) | (t1 >> 4), (t1 << 4) | (t2 >> 2), (t2 << 6) | t3];
                }
            },
            TrellisCodec::Vector4Restart64 => panic!("V4 rows are read in the package layout"),
        }
    }
}

/// The Mirai S readout (`I3S4Spec`) as a symmetric U4 matmul, with the factors of its 32-wide input Hadamard.
pub(super) fn load_readout<B: Backend>(
    context: &B::Context,
    tree: &ParameterTree<B>,
    spec: I3S4Spec,
    vocab_size: u32,
    model_dim: u32,
    data_type: DataType,
) -> Result<(LinearMatmul<B>, Allocation<B>), LinearMatmulError<B>> {
    if spec.layout != Layout::OutputInput
        || data_type != DataType::BF16
        || !model_dim.is_multiple_of(2 * READOUT_GROUP_SIZE)
    {
        return Err(LinearMatmulError::UnsupportedConfiguration(format!(
            "{spec:?} with {data_type:?} and model dim {model_dim}"
        )));
    }
    let (rows, columns) = (vocab_size, model_dim);
    let groups = columns / READOUT_GROUP_SIZE;
    let codes = tree.leaf("codes")?.validate(&[rows, columns * 3 / 8], DataType::U8)?.read_slice::<u8>()?;
    let row_scales = tree.leaf("row_scales")?.validate(&[rows], DataType::BF16)?.read_slice::<bf16>()?;
    let ladder_indices =
        tree.leaf("ladder_indices")?.validate(&[rows, groups / 2], DataType::U8)?.read_slice::<u8>()?;
    let ladder = tree.leaf("ladder")?.validate(&[16], DataType::F16)?.read_slice::<f16>()?;
    let padded_rows = QuantParams::new(QuantParamsLayout::GroupOutput, rows, groups).scale_shape()[1];
    let (u4_codes, scales) = repack_readout(&codes, &row_scales, &ladder_indices, &ladder, padded_rows as usize);
    let matrix = WeightMatrix::symmetric_u4(
        context.create_allocation_from_slice(&u4_codes).map_err(LinearMatmulError::BackendError)?,
        context.create_allocation_from_slice(&scales).map_err(LinearMatmulError::BackendError)?,
        rows,
        columns,
        READOUT_GROUP_SIZE,
    );
    let linear =
        LinearMatmul::from_matrix(context, matrix, None, None, columns, rows, data_type, data_type, data_type)?;
    let input_hadamard_factors =
        tree.leaf("input_hadamard_factors")?.validate(&[model_dim], DataType::I32)?.read_allocation()?;
    Ok((linear, input_hadamard_factors))
}

/// The readout's 3-bit codes c (packed LSB first) as U4 codes `level + 8` of the odd levels 2c - 7, and its group-major
/// bf16 scales `[columns / 64, padded_rows]` row_scale * ladder[index] (4-bit index, low nibble first; padding rows zero).
fn repack_readout(
    codes: &[u8],
    row_scales: &[bf16],
    ladder_indices: &[u8],
    ladder: &[f16],
    padded_rows: usize,
) -> (Vec<u8>, Vec<bf16>) {
    // 8 columns = 3 packed bytes = 4 output bytes
    let u4_codes = codes
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|&[b0, b1, b2]| {
            let packed = u32::from_le_bytes([b0, b1, b2, 0]);
            let nibble = |column: u32| (2 * ((packed >> (3 * column)) & 7) + 1) as u8;
            [0, 1, 2, 3].map(|pair| nibble(2 * pair) | (nibble(2 * pair + 1) << 4))
        })
        .collect();
    let rows = row_scales.len();
    let groups = 2 * ladder_indices.len() / rows;
    let scales = (0..groups)
        .flat_map(|group| {
            (0..padded_rows).map(move |row| {
                if row >= rows {
                    return bf16::ZERO;
                }
                let index = (ladder_indices[row * groups / 2 + group / 2] >> (4 * (group % 2))) & 15;
                bf16::from_f32(row_scales[row].to_f32() * ladder[index as usize].to_f32())
            })
        })
        .collect();
    (u4_codes, scales)
}

impl<B: Backend> Linear<B> for MiraiSLinear<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encoder.push_debug_group("mirai s linear");
        let rotated = self.transform.encode(&input, &self.signs, &self.mixing, batch_dim, encoder)?;
        let mut output = encoder.allocate_scratch_for_shape(&[batch_dim, self.output_dimension], DataType::BF16)?;
        let mut output_row_offset = 0;
        for part in &self.parts {
            part.projection.encode(
                ProjectionArguments {
                    input: &rotated,
                    codes: &part.codes,
                    row_scales: &part.row_scales,
                    codebook: &part.codebook,
                    rows: part.rows,
                    output: &mut output,
                    output_row_offset,
                    output_stride: self.output_dimension,
                },
                encoder,
            );
            output_row_offset += part.rows;
        }
        encoder.pop_debug_group();
        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../../unit/encodable_block/linear/mirai_s_test.rs"]
pub(crate) mod tests;
