use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        gpu_types::{hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE, trellis::COLUMN_GROUP_COUNT},
        kernel::{ActivationFormat, TrellisTransformKernel, matmul::MatmulA, mixing_dimension},
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{
            AnyIncoherenceProcessorSpec, HadamardSpec, HybridSpec, IncoherenceProcessingMode,
            IncoherenceProcessingMode::{Input, InputOutput},
        },
    },
    data_type::DataType,
    encodable_block::linear::{
        Gather, Linear, LinearBlockError, LinearEncodeError, LinearInput, LinearInputPreparation, LinearMatmul,
        input_rht::InputRht,
    },
    safetensors::ParameterTree,
};

enum Rotation<B: Backend> {
    Kronecker {
        kernel: <B::Kernels as Kernels>::TrellisTransformKernel,
        signs: B::GlobalGpuBuffer,
        orthogonal_factor: B::GlobalGpuBuffer,
    },
    Hadamard(InputRht<B>),
}

struct Group<B: Backend> {
    rotation: Rotation<B>,
    blocks: Box<[LinearMatmul<B>]>,
}

pub struct TransformedLinear<B: Backend> {
    groups: Box<[Group<B>]>,
    input_dim: u32,
    output_dim: u32,
    output_data_type: DataType,
}

use LinearBlockError::UnsupportedConfiguration as Unsupported;

async fn matrix_parts<'t, B: Backend>(
    spec: AnyWeightMatrixSpec,
    expected_rows: u32,
    tree: ParameterTree<'t, B>,
) -> Result<Vec<(u32, AnyWeightMatrixSpec, ParameterTree<'t, B>)>, LinearBlockError<B>> {
    if !matches!(spec, AnyWeightMatrixSpec::OutputStackSpec(_)) {
        return Ok(vec![(expected_rows, spec, tree)]);
    }
    let shape = tree.shape("part_rows")?;
    let part_rows = tree.read_slice::<i32>("part_rows", shape).await?;
    if part_rows.iter().any(|&rows| rows <= 0)
        || part_rows.iter().map(|&rows| i64::from(rows)).sum::<i64>() != i64::from(expected_rows)
    {
        return Err(Unsupported(format!("output-stack rows must be positive and sum to {expected_rows}")));
    }
    let parts = tree.subtree("parts");
    part_rows
        .iter()
        .enumerate()
        .map(|(index, &rows)| {
            let tree = parts.subtree(&index.to_string());
            Ok((rows as u32, tree.metadata("spec")?, tree))
        })
        .collect()
}

fn input_rht_mode(spec: &AnyWeightMatrixSpec) -> Option<IncoherenceProcessingMode> {
    match spec {
        AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            incoherence_spec:
                Some(AnyIncoherenceProcessorSpec::HadamardSpec(HadamardSpec {
                    block_size: HADAMARD_TRANSFORM_BLOCK_SIZE,
                    processing_mode: mode @ (IncoherenceProcessingMode::Input | IncoherenceProcessingMode::InputOutput),
                    ..
                })),
            ..
        }) => Some(mode.clone()),
        _ => None,
    }
}

impl<B: Backend> TransformedLinear<B> {
    pub(super) async fn new(
        input_dim: u32,
        output_dim: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        bias_tree: Option<&ParameterTree<'_, B>>,
        spec: AnyWeightMatrixSpec,
        weights_tree: ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, LinearBlockError<B>> {
        if bias_tree.is_some() && input_rht_mode(&spec).is_none() {
            return Err(Unsupported("biases on output-stacked or Kronecker linears".into()));
        }
        let parts = matrix_parts(spec, output_dim, weights_tree).await?;
        let in_place = parts.len() == 1;
        let mut groups = Vec::with_capacity(parts.len());
        for (rows, spec, tree) in parts {
            let AnyWeightMatrixSpec::HybridSpec(hybrid) = spec else {
                return Err(Unsupported(format!("{spec:?} as an output-stack part")));
            };
            let signs_tree = tree.subtree("incoherence_signs");
            let quantized_tree = tree.subtree("quantized");
            let quantization_spec = *hybrid.quantization_spec;
            let (rotation, blocks) = match hybrid.incoherence_spec {
                Some(AnyIncoherenceProcessorSpec::KroneckerSpec(spec)) if spec.processing_mode == Input => {
                    let mixing_dimension = mixing_dimension(input_dim);
                    let kernel = <B::Kernels as Kernels>::TrellisTransformKernel::new(context, input_dim)
                        .map_err(LinearBlockError::Backend)?;
                    let signs = signs_tree.read_buffer("signs", &[input_dim], DataType::I32).await?;
                    let orthogonal_factor = signs_tree
                        .read_buffer("orthogonal_factor", &[mixing_dimension, mixing_dimension], DataType::F32)
                        .await?;
                    let rotation = Rotation::Kronecker {
                        kernel,
                        signs,
                        orthogonal_factor,
                    };
                    let parts = matrix_parts(quantization_spec, rows, quantized_tree).await?;
                    let mut blocks = Vec::with_capacity(parts.len());
                    for (rows, spec, tree) in parts {
                        if !matches!(spec, AnyWeightMatrixSpec::QtipGaussianSpec(_)) {
                            return Err(Unsupported(format!("{spec:?} in a Kronecker linear")));
                        }
                        blocks.push(
                            LinearMatmul::load(
                                input_dim,
                                rows,
                                weights_data_type,
                                input_data_type,
                                output_data_type,
                                spec,
                                None,
                                &tree,
                                None,
                                context,
                            )
                            .await?,
                        );
                    }
                    (rotation, blocks.into_boxed_slice())
                },
                Some(AnyIncoherenceProcessorSpec::HadamardSpec(HadamardSpec {
                    block_size: HADAMARD_TRANSFORM_BLOCK_SIZE,
                    processing_mode: mode @ (Input | InputOutput),
                    ..
                })) if in_place => {
                    let (preparation, linear) = Self::load_hadamard(
                        input_dim,
                        rows,
                        weights_data_type,
                        input_data_type,
                        output_data_type,
                        mode,
                        true,
                        quantization_spec,
                        &tree,
                        bias_tree,
                        context,
                    )
                    .await?;
                    let rht = InputRht::new(input_data_type, preparation, in_place, context)
                        .map_err(LinearBlockError::Backend)?;
                    (Rotation::Hadamard(rht), Box::new([linear]) as Box<[_]>)
                },
                spec => return Err(Unsupported(format!("{spec:?} as an output-stack part"))),
            };
            groups.push(Group {
                rotation,
                blocks,
            });
        }
        Ok(Self {
            groups: groups.into_boxed_slice(),
            input_dim,
            output_dim,
            output_data_type,
        })
    }

    pub async fn new_with_rotated_output(
        input_dimension: u32,
        output_dimension: u32,
        data_type: DataType,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, LinearBlockError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        if input_rht_mode(&spec) != Some(InputOutput) {
            return Err(Unsupported(format!("{spec:?}")));
        }
        let quantization_spec = weights_tree.subtree("quantized").metadata("spec")?;
        let (input_preparation, inner_linear) = Self::load_hadamard(
            input_dimension,
            output_dimension,
            data_type,
            data_type,
            data_type,
            InputOutput,
            false,
            quantization_spec,
            &weights_tree,
            None,
            context,
        )
        .await?;
        Self::build_self_contained(data_type, input_preparation, inner_linear, context)
    }

    pub(super) async fn new_with_input_preparation(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        allow_prequantized_activation: bool,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Box<dyn Linear<B>>, Option<LinearInputPreparation<B>>), LinearBlockError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let Some(mode) = input_rht_mode(&spec) else {
            let linear = <dyn Linear<B>>::new_mixed_precision(
                input_dimension,
                output_dimension,
                weights_data_type,
                input_data_type,
                output_data_type,
                has_biases,
                parameter_tree,
                context,
            )
            .await?;
            return Ok((linear, None));
        };
        let quantization_spec = weights_tree.subtree("quantized").metadata("spec")?;
        let (input_preparation, inner_linear) = Self::load_hadamard(
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            mode,
            true,
            quantization_spec,
            &weights_tree,
            has_biases.then_some(parameter_tree),
            context,
        )
        .await?;
        if input_preparation.activation_format != ActivationFormat::Bf16 && !allow_prequantized_activation {
            let wrapper = Self::build_self_contained(input_data_type, input_preparation, inner_linear, context)?;
            Ok((Box::new(wrapper), None))
        } else {
            Ok((Box::new(inner_linear), Some(input_preparation)))
        }
    }

    async fn load_hadamard(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        mode: IncoherenceProcessingMode,
        restore_output_basis: bool,
        quantization_spec: AnyWeightMatrixSpec,
        weights_tree: &ParameterTree<'_, B>,
        bias_tree: Option<&ParameterTree<'_, B>>,
        context: &B::Context,
    ) -> Result<(LinearInputPreparation<B>, LinearMatmul<B>), LinearBlockError<B>> {
        let rht_signs =
            weights_tree.read_buffer("incoherence_signs.input_signs", &[input_dimension], DataType::I32).await?;
        let output_factors = if mode == IncoherenceProcessingMode::InputOutput && restore_output_basis {
            Some(weights_tree.read_buffer("incoherence_signs.output_signs", &[output_dimension], DataType::I32).await?)
        } else {
            if mode == IncoherenceProcessingMode::InputOutput {
                weights_tree.ignore("incoherence_signs.output_signs", &[output_dimension], DataType::I32)?;
            }
            None
        };
        let mut inner_linear = LinearMatmul::load(
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            quantization_spec,
            output_factors,
            &weights_tree.subtree("quantized"),
            bias_tree,
            context,
        )
        .await?;
        let activation_format = inner_linear.prepare_activation(context).map_err(LinearBlockError::Backend)?;
        Ok((
            LinearInputPreparation {
                rht_signs,
                activation_format,
            },
            inner_linear,
        ))
    }

    fn build_self_contained(
        input_data_type: DataType,
        input_preparation: LinearInputPreparation<B>,
        inner_linear: LinearMatmul<B>,
        context: &B::Context,
    ) -> Result<Self, LinearBlockError<B>> {
        let input_rht =
            InputRht::new(input_data_type, input_preparation, true, context).map_err(LinearBlockError::Backend)?;

        Ok(Self {
            input_dim: inner_linear.input_dim,
            output_dim: inner_linear.output_dim,
            output_data_type: inner_linear.output_data_type,
            groups: Box::new([Group {
                rotation: Rotation::Hadamard(input_rht),
                blocks: Box::new([inner_linear]),
            }]),
        })
    }

    fn prepare(
        &self,
        group: &Group<B>,
        input: impl BufferRef<Backend = B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<LinearInput<B>, B::Error> {
        let (kernel, signs, orthogonal_factor) = match &group.rotation {
            Rotation::Hadamard(rht) => {
                let format = group.blocks[0].select_activation_format(batch_dim, command_buffer.context());
                return rht.prepare(input, batch_dim, format, command_buffer);
            },
            Rotation::Kronecker {
                kernel,
                signs,
                orthogonal_factor,
            } => (kernel, signs, orthogonal_factor),
        };
        let mut values = command_buffer.allocate_scratch_for_shape(&[batch_dim, self.input_dim], DataType::I8)?;
        let mut column_group_sums =
            command_buffer.allocate_scratch_for_shape(&[batch_dim, COLUMN_GROUP_COUNT], DataType::F32)?;
        let mut scales = command_buffer.allocate_scratch_for_shape(&[batch_dim], DataType::F32)?;
        kernel.encode(
            input,
            signs,
            orthogonal_factor,
            &mut values,
            &mut column_group_sums,
            &mut scales,
            batch_dim,
            command_buffer,
        );
        Ok(MatmulA::Trellis {
            values,
            column_group_sums,
            scales,
        })
    }
}

impl<B: Backend> Linear<B> for TransformedLinear<B> {
    fn encode_input(
        &self,
        input: LinearInput<B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, LinearEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("linear (transformed)");

        if self.groups.len() == 1 && self.groups[0].blocks.len() == 1 {
            let group = &self.groups[0];
            let block = &group.blocks[0];
            let a = match input {
                MatmulA::FullPrecision {
                    values,
                    offset,
                } => match &group.rotation {
                    Rotation::Hadamard(rht) => {
                        let format = block.select_activation_format(batch_dim, command_buffer.context());
                        rht.prepare_in_place(values, offset, batch_dim, format, command_buffer)
                    },
                    Rotation::Kronecker {
                        ..
                    } => self.prepare(group, (&values).subrange(offset..), batch_dim, command_buffer),
                }
                .map_err(LinearEncodeError::Backend)?,
                prepared => prepared,
            };
            return block.encode_with_a(a.as_ref(), batch_dim, None::<Gather<&B::ScratchBuffer>>, command_buffer);
        }
        let MatmulA::FullPrecision {
            values: input,
            offset,
        } = input
        else {
            unreachable!();
        };
        let input = (&input).subrange(offset..);
        let mut output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.output_dim], self.output_data_type)
            .map_err(LinearEncodeError::Backend)?;
        let mut column = 0;
        for group in &self.groups {
            let a = self.prepare(group, input, batch_dim, command_buffer).map_err(LinearEncodeError::Backend)?;
            for block in &group.blocks {
                block.encode_columns(a.as_ref(), batch_dim, &mut output, column, self.output_dim, command_buffer)?;
                column += block.output_dim;
            }
        }
        Ok(output)
    }
}
