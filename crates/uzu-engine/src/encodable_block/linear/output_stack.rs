use std::{ops::Range, sync::Arc};

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding,
        gpu_types::hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
        kernel::{
            ActivationTransform,
            matmul::{ActivationFormat, MatmulA, MatmulGather},
        },
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{AnyIncoherenceProcessorSpec, HybridSpec, IncoherenceProcessingMode},
    },
    data_type::DataType,
    encodable_block::linear::{
        InputAccess, InputPreparationMode, InputRhtParameters, Linear, LinearBlockError, LinearEncodeError,
        LinearInput, LinearInputPreparation, LinearMatmul, OutputBasis,
        input_preparation::{InputPreparation, ScalarPreparation, TrellisPreparation},
    },
    safetensors::ParameterTree,
};

struct Part<B: Backend> {
    matrix: LinearMatmul<B>,
    rows: Range<u32>,
    preparation: InputPreparation<B>,
    rotation: Option<usize>,
}

pub enum OutputRows<I> {
    Prefix(u32),
    Gather {
        indices: I,
        count: u32,
    },
}

pub struct OutputStack<B: Backend> {
    parts: Box<[Part<B>]>,
    data_type: DataType,
    output_dim: u32,
    output_rotation: Option<(ActivationTransform<B>, B::GlobalGpuBuffer)>,
}

impl<B: Backend> OutputStack<B> {
    pub async fn load(
        input_dim: u32,
        output_dim: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        spec: AnyWeightMatrixSpec,
        tree: &ParameterTree<'_, B>,
        bias_tree: Option<&ParameterTree<'_, B>>,
        mode: InputPreparationMode,
        input_access: InputAccess,
        output_basis: OutputBasis,
        context: &B::Context,
    ) -> Result<(Self, Option<LinearInputPreparation<B>>), LinearBlockError<B>> {
        let keep_rotated = matches!(output_basis, OutputBasis::Rotated);
        if keep_rotated
            && !matches!(&spec,
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                incoherence_spec: Some(AnyIncoherenceProcessorSpec::HadamardSpec(hadamard)), ..
            }) if hadamard.processing_mode == IncoherenceProcessingMode::InputOutput)
        {
            return Err(LinearBlockError::UnsupportedConfiguration("expected input-output Hadamard transform".into()));
        }
        let mut loaded_rotations = Vec::new();
        let mut shared_rotation = None;
        let mut kronecker = None;
        let mut output_factors = None;
        let mut quantized_tree = None;
        let spec = match spec {
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                quantization_spec,
                incoherence_spec: Some(AnyIncoherenceProcessorSpec::KroneckerSpec(processor)),
                ..
            }) if processor.processing_mode == IncoherenceProcessingMode::Input => {
                kronecker = Some(Arc::new(
                    TrellisPreparation::load(
                        input_dim,
                        weights_data_type,
                        input_data_type,
                        output_data_type,
                        bias_tree.is_some(),
                        &quantization_spec,
                        tree,
                        context,
                    )
                    .await?,
                ));
                quantized_tree = Some(tree.subtree("quantized"));
                *quantization_spec
            },
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                quantization_spec,
                incoherence_spec: Some(AnyIncoherenceProcessorSpec::HadamardSpec(hadamard)),
                ..
            }) if matches!(
                hadamard.processing_mode,
                IncoherenceProcessingMode::Input | IncoherenceProcessingMode::InputOutput
            ) =>
            {
                let size = hadamard.block_size;
                if hadamard.processing_mode == IncoherenceProcessingMode::InputOutput {
                    if size != HADAMARD_TRANSFORM_BLOCK_SIZE {
                        return Err(LinearBlockError::UnsupportedConfiguration(format!(
                            "Hadamard block size {size} for input dimension {input_dim}"
                        )));
                    }
                    if keep_rotated {
                        tree.ignore("incoherence_signs.output_signs", &[output_dim], DataType::I32)?;
                    } else {
                        output_factors = Some(
                            tree.read_buffer("incoherence_signs.output_signs", &[output_dim], DataType::I32).await?,
                        );
                    }
                }
                shared_rotation = Some(InputRhtParameters::load(input_dim, size, tree, &mut loaded_rotations).await?);
                quantized_tree = Some(tree.subtree("quantized"));
                *quantization_spec
            },
            spec => spec,
        };
        let (specs, stacked) = match spec {
            AnyWeightMatrixSpec::OutputStackSpec(stack) => {
                (stack.into_parts(output_dim).map_err(LinearBlockError::UnsupportedConfiguration)?, true)
            },
            spec => (vec![(0..output_dim, spec)], false),
        };
        if stacked && bias_tree.is_some() {
            return Err(LinearBlockError::UnsupportedConfiguration("stacked linear biases".into()));
        }
        let output_rotation = if stacked {
            output_factors
                .take()
                .map(|signs| {
                    ActivationTransform::output_rht(
                        context,
                        output_data_type,
                        None,
                        HADAMARD_TRANSFORM_BLOCK_SIZE,
                        true,
                    )
                    .map(|kernel| (kernel, signs))
                })
                .transpose()
                .map_err(LinearBlockError::Backend)?
        } else {
            None
        };
        let tree = quantized_tree.as_ref().unwrap_or(tree);
        let mode = if stacked {
            InputPreparationMode::InLinear
        } else {
            mode
        };
        let mut external = None;
        let mut parts = Vec::with_capacity(specs.len());
        for (index, (rows, spec)) in specs.into_iter().enumerate() {
            let part_tree = stacked.then(|| tree.subtree(&format!("parts.{index}")));
            let tree = part_tree.as_ref().unwrap_or(tree);
            let (spec, hadamard_size) = match spec {
                AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                    quantization_spec,
                    incoherence_spec: Some(AnyIncoherenceProcessorSpec::HadamardSpec(hadamard)),
                    ..
                }) if hadamard.processing_mode == IncoherenceProcessingMode::Input => {
                    (*quantization_spec, Some(hadamard.block_size))
                },
                spec => (spec, None),
            };
            let rotation = if let Some(size) = hadamard_size {
                if shared_rotation.is_some() {
                    return Err(LinearBlockError::UnsupportedConfiguration(
                        "part input Hadamard cannot be combined with shared rotation".into(),
                    ));
                }
                Some(InputRhtParameters::load(input_dim, size, tree, &mut loaded_rotations).await?)
            } else {
                shared_rotation
            };
            let rht = rotation.map(|index| &loaded_rotations[index].0);
            let quantized_tree = hadamard_size.map(|_| tree.subtree("quantized"));
            let tree = quantized_tree.as_ref().unwrap_or(tree);
            let mut matrix = LinearMatmul::load(
                input_dim,
                rows.end - rows.start,
                weights_data_type,
                input_data_type,
                output_data_type,
                spec,
                output_factors.take(),
                tree,
                bias_tree,
                context,
            )
            .await?;
            let format = matrix.prepare_activation(context).map_err(LinearBlockError::Backend)?;
            let use_external = rht.is_some()
                && matches!(mode,
                    InputPreparationMode::AllowExternal { allow_quantized_activations }
                        if allow_quantized_activations || format == ActivationFormat::Bf16);
            let (rht, format) = if use_external {
                external = rht.map(|rht| LinearInputPreparation {
                    rht: InputRhtParameters {
                        hadamard_size: rht.hadamard_size,
                        signs: rht.signs.clone(),
                    },
                    activation_format: format,
                });
                (None, ActivationFormat::Bf16)
            } else {
                (rht, format)
            };
            let preparation = if let Some(trellis) = &kronecker {
                InputPreparation::Trellis(Arc::clone(trellis))
            } else {
                ScalarPreparation::new(
                    input_dim,
                    input_data_type,
                    rht,
                    format,
                    !stacked && matches!(input_access, InputAccess::MayOverwrite),
                    context,
                )
                .map(InputPreparation::Scalar)
                .map_err(LinearBlockError::Backend)?
            };
            parts.push(Part {
                matrix,
                rows,
                preparation,
                rotation,
            });
        }
        Ok((
            Self {
                parts: parts.into_boxed_slice(),
                data_type: output_data_type,
                output_dim,
                output_rotation,
            },
            external,
        ))
    }
}

impl<B: Backend> OutputStack<B> {
    pub fn encode_input(
        &self,
        input: MatmulA<impl BufferRef<Backend = B>>,
        batch_dim: u32,
        rows: OutputRows<impl BufferRef<Backend = B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>> {
        let output_dim = match rows {
            OutputRows::Prefix(count)
            | OutputRows::Gather {
                count,
                ..
            } => count,
        };
        let mut output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, output_dim], self.data_type)
            .map_err(LinearEncodeError::Backend)?;
        let mut prepared: Vec<(Option<usize>, ActivationFormat, LinearInput<B>)> = Vec::new();
        for part in &self.parts {
            let (column, column_count, gather) = match rows {
                OutputRows::Gather {
                    indices,
                    ..
                } => (
                    0,
                    output_dim,
                    Some(MatmulGather {
                        indices,
                        first_row: part.rows.start,
                        row_count: part.rows.end - part.rows.start,
                    }),
                ),
                OutputRows::Prefix(_) if part.rows.start < output_dim => {
                    (part.rows.start, part.rows.end.min(output_dim) - part.rows.start, None)
                },
                OutputRows::Prefix(_) => break,
            };
            let format = part.preparation.select_format(ActivationFormat::Trellis, || {
                if gather.is_none() && !part.preparation.is_identity() && matches!(input, MatmulA::FullPrecision { .. })
                {
                    part.matrix.select_activation_format(batch_dim, Some(column_count), command_buffer.context())
                } else {
                    ActivationFormat::Bf16
                }
            });
            let index = if let MatmulA::FullPrecision {
                values,
                offset,
            } = input
            {
                if let Some(index) =
                    prepared.iter().position(|(rotation, f, _)| *rotation == part.rotation && *f == format)
                {
                    Some(index)
                } else if let Some(values) = part
                    .preparation
                    .prepare(values.subrange(offset..), batch_dim, format, command_buffer)
                    .map_err(LinearEncodeError::Backend)?
                {
                    prepared.push((part.rotation, format, values));
                    Some(prepared.len() - 1)
                } else {
                    None
                }
            } else if part.preparation.requires_full_precision() {
                return Err(LinearEncodeError::UnsupportedInput(
                    "stacked input rotations require full-precision input",
                ));
            } else {
                None
            };
            if let Some(index) = index {
                part.matrix.encode_columns(
                    prepared[index].2.as_ref(),
                    batch_dim,
                    &mut output,
                    column,
                    column_count,
                    output_dim,
                    gather,
                    command_buffer,
                )?;
            } else {
                part.matrix.encode_columns(
                    input,
                    batch_dim,
                    &mut output,
                    column,
                    column_count,
                    output_dim,
                    gather,
                    command_buffer,
                )?;
            }
        }
        self.rotate_output(&mut output, batch_dim, command_buffer);
        Ok(output)
    }

    fn rotate_output(
        &self,
        output: &mut B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        if let Some((transform, signs)) = &self.output_rotation {
            transform.encode_fp_in_place(output, Some(signs), None, batch_dim, self.output_dim, command_buffer);
        }
    }
}

impl<B: Backend> Linear<B> for OutputStack<B> {
    fn encode_input(
        &self,
        input: LinearInput<B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>> {
        if let [part] = &*self.parts
            && part.preparation.supports_owned_input()
        {
            let input = match input {
                MatmulA::FullPrecision {
                    values,
                    offset,
                } => {
                    let format = part.preparation.select_format(ActivationFormat::Trellis, || {
                        part.matrix.select_activation_format(batch_dim, None, command_buffer.context())
                    });
                    part.preparation
                        .prepare_owned(values, offset, batch_dim, format, command_buffer)
                        .map_err(LinearEncodeError::Backend)?
                },
                prepared => prepared,
            };
            let mut output = part.matrix.encode_input(input, batch_dim, command_buffer)?;
            self.rotate_output(&mut output, batch_dim, command_buffer);
            Ok(output)
        } else {
            OutputStack::encode_input(
                self,
                input.as_ref(),
                batch_dim,
                OutputRows::<&B::ScratchBuffer>::Prefix(self.output_dim),
                command_buffer,
            )
        }
    }

    fn select_activation_format(
        &self,
        batch_dim: u32,
        output_dim: Option<u32>,
        context: &B::Context,
    ) -> ActivationFormat {
        match &*self.parts {
            [part] => part.preparation.select_format(ActivationFormat::Bf16, || {
                part.matrix.select_activation_format(batch_dim, output_dim, context)
            }),
            _ => ActivationFormat::Bf16,
        }
    }
}
