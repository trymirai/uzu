mod dense;
mod moe;

use std::collections::BTreeMap;

use half::{bf16, f16};
use serde::Deserialize;
use thiserror::Error;

pub use dense::DenseMlp;
pub use moe::{MoeBlock, MoeBlockError};

use super::linear::{FullPrecisionLinear, Linear, LinearBlockError};
use crate::{
    ArrayElement, DataType,
    backends::common::{Backend, CommandBuffer, kernel::mlp_gate_act_mul::MlpGateActMulEncodable},
    config::{LinearConfig, MLPConfig},
    encodable_block::EncodingParameters,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
    utils::env_utils::EnvVar,
};

const MLP_BLOCK_SIZE: usize = 32;

pub trait Mlp<B: Backend> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error>;
}

#[derive(Debug, Error)]
pub enum MlpBlockError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Linear block error: {0}")]
    LinearBlockError(#[from] LinearBlockError<B>),
    #[error("MoeBlock error: {0}")]
    MoeBlockError(#[from] MoeBlockError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
}

impl<B: Backend> dyn Mlp<B> {
    pub fn new(
        config: &MLPConfig,
        model_dimension: usize,
        hidden_dimension: usize,
        context: &B::Context,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Box<dyn Mlp<B>>, MlpBlockError<B>> {
        if let MLPConfig::Dense(dense_config) = config {
            let data_type: DataType = dense_config.linear_config.activation_precision().into();
            let label = parameter_tree.path_prefix().expect("Dense MLP should have a parameter path");
            let selected_blocks = static_block_indices_from_env(&label, hidden_dimension)
                .map(Ok)
                .or_else(|| {
                    static_keep_ratio_from_env()
                        .map(|keep_ratio| select_static_mlp_blocks(parameter_tree, hidden_dimension, keep_ratio))
                })
                .transpose()?;

            let prefill_reduced = selected_blocks
                .as_deref()
                .map(|selected_blocks| {
                    let LinearConfig::FullPrecision {
                        precision,
                    } = &dense_config.linear_config
                    else {
                        panic!("Reduced-width MLP currently requires full-precision linears");
                    };
                    let selected_hidden = selected_hidden_indices(selected_blocks);
                    let selected_up = selected_up_rows(&selected_hidden, hidden_dimension);
                    let reduced_hidden_dimension = selected_hidden.len();
                    let up_projection = FullPrecisionLinear::new_selected_rows(
                        context,
                        (*precision).into(),
                        model_dimension,
                        2 * hidden_dimension,
                        &selected_up,
                        &parameter_tree.subtree("up_projection")?,
                        ArrayId::Main,
                        ArrayId::MlpFusedUp,
                    )
                    .map_err(LinearBlockError::FullPrecisionLinearError)?;
                    let gate = MlpGateActMulEncodable::new(
                        context,
                        data_type,
                        dense_config.activation.clone(),
                        reduced_hidden_dimension,
                    )
                    .map_err(MlpBlockError::BackendError)?;
                    let down_projection = FullPrecisionLinear::new_selected_columns(
                        context,
                        (*precision).into(),
                        hidden_dimension,
                        model_dimension,
                        &selected_hidden,
                        &parameter_tree.subtree("down_projection")?,
                        ArrayId::MlpHidden,
                        ArrayId::Main,
                    )
                    .map_err(LinearBlockError::FullPrecisionLinearError)?;
                    Ok::<_, MlpBlockError<B>>((
                        Box::new(up_projection) as Box<dyn Linear<B>>,
                        gate,
                        Box::new(down_projection) as Box<dyn Linear<B>>,
                    ))
                })
                .transpose()?;

            let up_projection = <dyn Linear<B>>::new(
                &dense_config.linear_config,
                false,
                model_dimension,
                [2 * hidden_dimension],
                context,
                &parameter_tree.subtree("up_projection")?,
                ArrayId::Main,
                ArrayId::MlpFusedUp,
            )?;
            let gate_activation =
                MlpGateActMulEncodable::new(context, data_type, dense_config.activation.clone(), hidden_dimension)
                    .map_err(MlpBlockError::BackendError)?;
            let down_projection = <dyn Linear<B>>::new(
                &dense_config.linear_config,
                false,
                hidden_dimension,
                [model_dimension],
                context,
                &parameter_tree.subtree("down_projection")?,
                ArrayId::MlpHidden,
                ArrayId::Main,
            )?;
            let (prefill_reduced_up, prefill_reduced_gate, prefill_reduced_down) = match prefill_reduced {
                Some((up, gate, down)) => (Some(up), Some(gate), Some(down)),
                None => (None, None, None),
            };

            return Ok(Box::new(DenseMlp::new(
                up_projection,
                gate_activation,
                down_projection,
                prefill_reduced_up,
                prefill_reduced_gate,
                prefill_reduced_down,
            )));
        }

        if let MLPConfig::MixtureOfExperts(mixture_of_experts_config) = config {
            let mixture_of_experts_block =
                MoeBlock::new(context, mixture_of_experts_config, model_dimension, hidden_dimension, parameter_tree)?;
            return Ok(Box::new(mixture_of_experts_block));
        }

        unreachable!("Unknown MLP config")
    }
}

fn static_keep_ratio_from_env() -> Option<f32> {
    let value = EnvVar::MlpStaticKeepRatio.value();
    if value.is_empty() {
        return None;
    }
    let keep_ratio = value.parse::<f32>().expect("UZU_MLP_STATIC_KEEP_RATIO must be a float");
    assert!((0.0..=1.0).contains(&keep_ratio) && keep_ratio > 0.0, "UZU_MLP_STATIC_KEEP_RATIO must be in (0, 1]");
    Some(keep_ratio)
}

fn static_block_indices_from_env(
    label: &str,
    hidden_dimension: usize,
) -> Option<Box<[usize]>> {
    block_indices_from_layer_json(label, hidden_dimension).or_else(|| {
        let value = EnvVar::MlpStaticBlocks.value();
        if value.is_empty() {
            return None;
        }
        Some(parse_block_indices(&value, hidden_dimension / MLP_BLOCK_SIZE, EnvVar::MlpStaticBlocks.key()))
    })
}

pub(super) fn block_indices_from_layer_json(
    label: &str,
    hidden_dimension: usize,
) -> Option<Box<[usize]>> {
    assert_eq!(hidden_dimension % MLP_BLOCK_SIZE, 0, "MLP hidden width must divide block size");
    let value = EnvVar::MlpBlocksByLayerJson.value();
    if value.is_empty() {
        return None;
    }
    let blocks = serde_json::from_str::<LayerBlockMap>(&value)
        .unwrap_or_else(|_| panic!("{} must be valid JSON", EnvVar::MlpBlocksByLayerJson.key()));
    let layer_index = layer_index_from_label(label).expect("MLP label must include a layer index");
    let block_count = hidden_dimension / MLP_BLOCK_SIZE;
    let layer_key = layer_index.to_string();
    blocks
        .0
        .get(&layer_key)
        .or_else(|| blocks.0.get(label))
        .map(|selected| parse_block_indices_vec(selected.clone(), block_count, EnvVar::MlpBlocksByLayerJson.key()))
}

fn layer_index_from_label(label: &str) -> Option<usize> {
    let parts = label.split('.').collect::<Vec<_>>();
    parts
        .windows(2)
        .find_map(|window| matches!(window, ["layer" | "layers", _]).then(|| window[1].parse::<usize>().ok()))
        .flatten()
}

fn parse_block_indices(
    value: &str,
    block_count: usize,
    key: &str,
) -> Box<[usize]> {
    parse_block_indices_vec(
        value
            .split(',')
            .map(|item| item.trim().parse::<usize>().unwrap_or_else(|_| panic!("{key} must contain integers")))
            .collect(),
        block_count,
        key,
    )
}

fn parse_block_indices_vec(
    mut selected: Vec<usize>,
    block_count: usize,
    key: &str,
) -> Box<[usize]> {
    assert!(!selected.is_empty(), "{key} cannot be empty");
    selected.sort_unstable();
    selected.dedup();
    assert!(selected.iter().all(|&block| block < block_count), "{key} must contain valid block indices");
    selected.into_boxed_slice()
}

#[derive(Deserialize)]
struct LayerBlockMap(BTreeMap<String, Vec<usize>>);

fn select_static_mlp_blocks<B: Backend>(
    parameter_tree: &ParameterTree<B::Context>,
    hidden_dimension: usize,
    keep_ratio: f32,
) -> Result<Box<[usize]>, MlpBlockError<B>> {
    assert_eq!(hidden_dimension % MLP_BLOCK_SIZE, 0, "MLP hidden width must divide block size");
    let up_weights = parameter_tree.subtree("up_projection")?.leaf_array("weights")?;
    let down_weights = parameter_tree.subtree("down_projection")?.leaf_array("weights")?;
    let block_scores = match up_weights.data_type() {
        DataType::BF16 => static_block_scores_t::<B, bf16>(&up_weights, &down_weights, hidden_dimension),
        DataType::F16 => static_block_scores_t::<B, f16>(&up_weights, &down_weights, hidden_dimension),
        DataType::F32 => static_block_scores_t::<B, f32>(&up_weights, &down_weights, hidden_dimension),
        dtype => panic!("Unsupported MLP dtype for static reduced-width selector: {dtype:?}"),
    };
    let keep_blocks = ((block_scores.len() as f32) * keep_ratio).ceil() as usize;
    let mut ranked = (0..block_scores.len()).collect::<Vec<_>>();
    ranked.sort_unstable_by(|&lhs, &rhs| {
        block_scores[rhs].partial_cmp(&block_scores[lhs]).expect("MLP selector scores must be finite")
    });
    let mut selected = ranked.into_iter().take(keep_blocks).collect::<Vec<_>>();
    selected.sort_unstable();
    Ok(selected.into_boxed_slice())
}

fn static_block_scores_t<B: Backend, T: ArrayElement + Copy>(
    up_weights: &crate::array::Array<B>,
    down_weights: &crate::array::Array<B>,
    hidden_dimension: usize,
) -> Box<[f32]>
where
    f32: From<T>,
{
    let up = up_weights.as_slice::<T>();
    let down = down_weights.as_slice::<T>();
    let input_dim = up_weights.shape()[1];
    let output_dim = down_weights.shape()[0];
    let block_count = hidden_dimension / MLP_BLOCK_SIZE;
    let mut scores = vec![0.0_f32; block_count];

    for (block_index, score) in scores.iter_mut().enumerate() {
        let block_start = block_index * MLP_BLOCK_SIZE;
        let mut up_energy = 0.0_f32;
        let mut gate_energy = 0.0_f32;
        for row in block_start..block_start + MLP_BLOCK_SIZE {
            let up_row_start = row * input_dim;
            let gate_row_start = (hidden_dimension + row) * input_dim;
            up_energy += up[up_row_start..up_row_start + input_dim]
                .iter()
                .map(|value| {
                    let value = f32::from(*value);
                    value * value
                })
                .sum::<f32>();
            gate_energy += up[gate_row_start..gate_row_start + input_dim]
                .iter()
                .map(|value| {
                    let value = f32::from(*value);
                    value * value
                })
                .sum::<f32>();
        }

        let mut down_energy = 0.0_f32;
        for row in 0..output_dim {
            let row_start = row * hidden_dimension + block_start;
            down_energy += down[row_start..row_start + MLP_BLOCK_SIZE]
                .iter()
                .map(|value| {
                    let value = f32::from(*value);
                    value * value
                })
                .sum::<f32>();
        }

        *score = up_energy * gate_energy * down_energy.sqrt();
    }

    scores.into_boxed_slice()
}

fn selected_hidden_indices(selected_blocks: &[usize]) -> Box<[usize]> {
    selected_blocks
        .iter()
        .flat_map(|&block| (block * MLP_BLOCK_SIZE)..((block + 1) * MLP_BLOCK_SIZE))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn selected_up_rows(
    selected_hidden: &[usize],
    hidden_dimension: usize,
) -> Box<[usize]> {
    selected_hidden
        .iter()
        .copied()
        .chain(selected_hidden.iter().map(|&index| hidden_dimension + index))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::{MLP_BLOCK_SIZE, selected_hidden_indices, selected_up_rows};

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn matmul(
        input: &[f32],
        rows: usize,
        cols: usize,
        weights: &[f32],
        out_cols: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0; rows * out_cols];
        for row in 0..rows {
            for col in 0..out_cols {
                let mut acc = 0.0;
                for k in 0..cols {
                    acc += input[row * cols + k] * weights[col * cols + k];
                }
                out[row * out_cols + col] = acc;
            }
        }
        out
    }

    fn compact_rows(
        weights: &[f32],
        input_dim: usize,
        selected_rows: &[usize],
    ) -> Vec<f32> {
        let mut compact = Vec::with_capacity(selected_rows.len() * input_dim);
        for &row in selected_rows {
            compact.extend_from_slice(&weights[row * input_dim..(row + 1) * input_dim]);
        }
        compact
    }

    fn compact_columns(
        weights: &[f32],
        input_dim: usize,
        output_dim: usize,
        selected_columns: &[usize],
    ) -> Vec<f32> {
        let mut compact = Vec::with_capacity(output_dim * selected_columns.len());
        for row in 0..output_dim {
            for &column in selected_columns {
                compact.push(weights[row * input_dim + column]);
            }
        }
        compact
    }

    #[test]
    fn reduced_width_selection_matches_masked_dense_mlp() {
        let batch = 2;
        let input_dim = 3;
        let hidden_dim = MLP_BLOCK_SIZE * 2;
        let output_dim = 4;
        let input = vec![0.5, -0.25, 1.0, -1.0, 0.75, 0.25];
        let up_weights = (0..(2 * hidden_dim * input_dim)).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect::<Vec<_>>();
        let down_weights = (0..(output_dim * hidden_dim)).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect::<Vec<_>>();
        let selected_blocks = [1_usize];
        let selected_hidden = selected_hidden_indices(&selected_blocks);
        let selected_up = selected_up_rows(&selected_hidden, hidden_dim);

        let fused = matmul(&input, batch, input_dim, &up_weights, 2 * hidden_dim);
        let mut hidden = vec![0.0; batch * hidden_dim];
        for row in 0..batch {
            for j in 0..hidden_dim {
                hidden[row * hidden_dim + j] =
                    fused[row * 2 * hidden_dim + j] * silu(fused[row * 2 * hidden_dim + hidden_dim + j]);
            }
        }
        let dense_output = matmul(&hidden, batch, hidden_dim, &down_weights, output_dim);

        let reduced_up = compact_rows(&up_weights, input_dim, &selected_up);
        let reduced_fused = matmul(&input, batch, input_dim, &reduced_up, selected_up.len());
        let mut reduced_hidden = vec![0.0; batch * selected_hidden.len()];
        for row in 0..batch {
            for (index, _) in selected_hidden.iter().enumerate() {
                reduced_hidden[row * selected_hidden.len() + index] = reduced_fused[row * selected_up.len() + index]
                    * silu(reduced_fused[row * selected_up.len() + selected_hidden.len() + index]);
            }
        }
        let reduced_down = compact_columns(&down_weights, hidden_dim, output_dim, &selected_hidden);
        let reduced_output = matmul(&reduced_hidden, batch, selected_hidden.len(), &reduced_down, output_dim);

        let mut masked_output = vec![0.0; batch * output_dim];
        for row in 0..batch {
            for out in 0..output_dim {
                let mut acc = 0.0;
                for &column in selected_hidden.iter() {
                    acc += hidden[row * hidden_dim + column] * down_weights[out * hidden_dim + column];
                }
                masked_output[row * output_dim + out] = acc;
            }
        }

        for (lhs, rhs) in reduced_output.iter().zip(masked_output.iter()) {
            assert!((lhs - rhs).abs() < 1e-5, "reduced output mismatch: {lhs} vs {rhs}");
        }
        assert!(dense_output.iter().zip(masked_output.iter()).any(|(lhs, rhs)| (lhs - rhs).abs() > 1e-4));
    }
}
