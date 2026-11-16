use std::sync::Arc;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding,
        gpu_types::hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
        kernel::matmul::{ActivationFormat, MatmulA, MatmulGather},
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{AnyIncoherenceProcessorSpec, HadamardSpec, HybridSpec, IncoherenceProcessingMode},
    },
    data_type::DataType,
    encodable_block::linear::{
        InputRhtParameters, Linear, LinearEncodeError, LinearInputPreparation, LinearMatmul, LinearMatmulError,
        input_rht::InputRht,
    },
    safetensors::ParameterTree,
};

pub struct UntiedReadout<B: Backend> {
    parts: Box<[(Option<InputRht<B>>, LinearMatmul<B>)]>,
    data_type: DataType,
}

impl<B: Backend> UntiedReadout<B> {
    pub async fn load(
        vocab_size: u32,
        model_dim: u32,
        data_type: DataType,
        spec: AnyWeightMatrixSpec,
        tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, LinearMatmulError<B>> {
        let load_part = async |rows, spec, tree: &ParameterTree<'_, B>| -> Result<_, LinearMatmulError<B>> {
            let AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                quantization_spec,
                incoherence_spec:
                    Some(AnyIncoherenceProcessorSpec::HadamardSpec(HadamardSpec {
                        block_size: hadamard_size @ (HADAMARD_TRANSFORM_BLOCK_SIZE | 128),
                        processing_mode: IncoherenceProcessingMode::Input,
                        ..
                    })),
                ..
            }) = spec
            else {
                let linear = LinearMatmul::load(
                    model_dim, rows, data_type, data_type, data_type, spec, None, tree, None, context,
                )
                .await?;
                return Ok((None, linear));
            };
            if !model_dim.is_multiple_of(hadamard_size) {
                return Err(LinearMatmulError::UnsupportedConfiguration(format!(
                    "Hadamard block size {hadamard_size} for input dimension {model_dim}"
                )));
            }

            let mut linear = LinearMatmul::load(
                model_dim,
                rows,
                data_type,
                data_type,
                data_type,
                *quantization_spec,
                None,
                &tree.subtree("quantized"),
                None,
                context,
            )
            .await?;
            let signs = Arc::new(
                tree.subtree("incoherence_signs").read_buffer("input_signs", &[model_dim], DataType::I32).await?,
            );
            let activation_format = linear.prepare_activation(context).map_err(LinearMatmulError::BackendError)?;
            let preparation = LinearInputPreparation {
                rht: InputRhtParameters {
                    hadamard_size,
                    signs,
                },
                activation_format,
            };
            let input_rht = InputRht::new(data_type, preparation, /* in_place */ false, context)
                .map_err(LinearMatmulError::BackendError)?;
            Ok((Some(input_rht), linear))
        };

        let parts = match spec {
            AnyWeightMatrixSpec::OutputStackSpec(stack) => {
                let stack_rows = stack.parts.iter().map(|part| part.rows).sum::<u32>();
                if stack_rows != vocab_size {
                    return Err(LinearMatmulError::UnsupportedConfiguration(format!(
                        "stacked readout has {stack_rows} rows, expected {vocab_size}"
                    )));
                }
                let mut parts = Vec::with_capacity(stack.parts.len());
                for (index, part) in stack.parts.into_iter().enumerate() {
                    parts.push(load_part(part.rows, part.spec, &tree.subtree(&format!("parts.{index}"))).await?);
                }
                parts.into_boxed_slice()
            },
            spec => Box::new([load_part(vocab_size, spec, tree).await?]) as Box<[_]>,
        };
        Ok(Self {
            parts,
            data_type,
        })
    }

    pub fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        batch_dim: u32,
        output_dim: u32,
        gather_indices: Option<impl BufferRef<Backend = B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>> {
        let mut output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, output_dim], self.data_type)
            .map_err(LinearEncodeError::Backend)?;
        let mut first_row = 0;
        for (input_rht, linear) in &self.parts {
            let row_count = linear.output_dim();
            let (column, column_count, gather) = match gather_indices {
                Some(indices) => (
                    0,
                    output_dim,
                    Some(MatmulGather {
                        indices,
                        first_row,
                        row_count,
                    }),
                ),
                None if first_row < output_dim => (first_row, row_count.min(output_dim - first_row), None),
                None => break,
            };
            match input_rht {
                Some(input_rht) => {
                    let format = if gather.is_some() {
                        ActivationFormat::Bf16
                    } else {
                        linear.select_activation_format(batch_dim, Some(column_count), command_buffer.context())
                    };
                    let input = input_rht
                        .prepare(input, batch_dim, format, command_buffer)
                        .map_err(LinearEncodeError::Backend)?;
                    linear.encode_columns(
                        input.as_ref(),
                        batch_dim,
                        &mut output,
                        column,
                        column_count,
                        output_dim,
                        gather,
                        command_buffer,
                    )?;
                },
                None => linear.encode_columns(
                    MatmulA::full_precision(input),
                    batch_dim,
                    &mut output,
                    column,
                    column_count,
                    output_dim,
                    gather,
                    command_buffer,
                )?,
            }
            first_row += row_count;
        }
        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../../unit/encodable_block/untied_readout_test.rs"]
mod tests;
