use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        gpu_types::{
            hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
            trellis::{COLUMN_GROUP_COUNT, MAX_DIMENSION, MAX_HADAMARD_SIZE, MIN_HADAMARD_SIZE},
        },
        kernel::{ActivationFormat, TrellisTransformKernel, matmul::MatmulA, mixing_dimension},
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{
            AnyIncoherenceProcessorSpec, HadamardSpec, HybridSpec,
            IncoherenceProcessingMode::{Input, InputOutput},
        },
    },
    data_type::DataType,
    encodable_block::linear::{
        Linear, LinearBlockError, LinearEncodeError, LinearInput, LinearInputPreparation, LinearMatmul,
        input_rht::InputRht,
    },
    safetensors::ParameterTree,
};

pub struct TransformedLinear<B: Backend> {
    transform: Transform<B>,
}

enum Transform<B: Backend> {
    Hadamard {
        input_rht: InputRht<B>,
        matrix: LinearMatmul<B>,
    },
    Kronecker {
        kernel: <B::Kernels as Kernels>::TrellisTransformKernel,
        signs: B::GlobalGpuBuffer,
        orthogonal_factor: B::GlobalGpuBuffer,
        matrices: Box<[LinearMatmul<B>]>,
        input_dimension: u32,
        output_dimension: u32,
    },
}

pub(super) enum InputPreparationMode {
    InLinear,
    AllowExternal {
        allow_quantized_activations: bool,
    },
}

impl<B: Backend> TransformedLinear<B> {
    pub(super) async fn new(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        bias_tree: Option<&ParameterTree<'_, B>>,
        spec: HybridSpec,
        weights_tree: &ParameterTree<'_, B>,
        input_preparation_mode: InputPreparationMode,
        context: &B::Context,
    ) -> Result<(Box<dyn Linear<B>>, Option<LinearInputPreparation<B>>), LinearBlockError<B>> {
        match spec.incoherence_spec {
            Some(AnyIncoherenceProcessorSpec::HadamardSpec(hadamard))
                if matches!(hadamard.processing_mode, Input | InputOutput) =>
            {
                let (preparation, matrix) = Self::load_hadamard(
                    input_dimension,
                    output_dimension,
                    weights_data_type,
                    input_data_type,
                    output_data_type,
                    hadamard,
                    *spec.quantization_spec,
                    true,
                    weights_tree,
                    bias_tree,
                    context,
                )
                .await?;
                let use_external_preparation = match input_preparation_mode {
                    InputPreparationMode::InLinear => false,
                    InputPreparationMode::AllowExternal {
                        allow_quantized_activations,
                    } => preparation.activation_format == ActivationFormat::Bf16 || allow_quantized_activations,
                };
                if use_external_preparation {
                    Ok((Box::new(matrix), Some(preparation)))
                } else {
                    let linear = Self::build_self_contained(input_data_type, preparation, matrix, context)?;
                    Ok((Box::new(linear), None))
                }
            },
            Some(AnyIncoherenceProcessorSpec::KroneckerSpec(kronecker)) if kronecker.processing_mode == Input => {
                let linear = Self::load_kronecker(
                    input_dimension,
                    output_dimension,
                    weights_data_type,
                    input_data_type,
                    output_data_type,
                    bias_tree,
                    *spec.quantization_spec,
                    weights_tree,
                    context,
                )
                .await?;
                Ok((Box::new(linear), None))
            },
            spec => Err(LinearBlockError::UnsupportedConfiguration(format!("{spec:?}"))),
        }
    }

    async fn load_kronecker(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        bias_tree: Option<&ParameterTree<'_, B>>,
        spec: AnyWeightMatrixSpec,
        weights_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, LinearBlockError<B>> {
        if bias_tree.is_some() {
            return Err(LinearBlockError::UnsupportedConfiguration("Kronecker linear biases".into()));
        }
        if [weights_data_type, input_data_type, output_data_type] != [DataType::BF16; 3] {
            return Err(LinearBlockError::UnsupportedConfiguration("Kronecker linears require BF16".into()));
        }
        let hadamard_size = input_dimension.isolate_lowest_one();
        if !(MIN_HADAMARD_SIZE..=MAX_HADAMARD_SIZE).contains(&hadamard_size) || input_dimension > MAX_DIMENSION {
            return Err(LinearBlockError::UnsupportedConfiguration(format!(
                "Kronecker rotation over {input_dimension} columns"
            )));
        }
        let kernel = <B::Kernels as Kernels>::TrellisTransformKernel::new(context, input_dimension)
            .map_err(LinearBlockError::Backend)?;
        let signs = weights_tree.read_buffer("incoherence_signs.signs", &[input_dimension], DataType::I32).await?;
        let mixing_dimension = mixing_dimension(input_dimension);
        let orthogonal_factor = weights_tree
            .read_buffer("incoherence_signs.orthogonal_factor", &[mixing_dimension, mixing_dimension], DataType::F32)
            .await?;
        let load_matrix = async |rows, spec, tree: &ParameterTree<'_, B>| {
            if !matches!(spec, AnyWeightMatrixSpec::QtipGaussianSpec(_)) {
                return Err(LinearBlockError::UnsupportedConfiguration("expected Trellis weights".into()));
            }
            LinearMatmul::load(
                input_dimension,
                rows,
                DataType::BF16,
                DataType::BF16,
                DataType::BF16,
                spec,
                None,
                tree,
                None,
                context,
            )
            .await
            .map_err(LinearBlockError::from)
        };
        let matrices = match spec {
            AnyWeightMatrixSpec::OutputStackSpec(_) => {
                let shape = weights_tree.shape("quantized.part_rows")?;
                if shape.len() != 1 {
                    return Err(LinearBlockError::UnsupportedConfiguration(format!(
                        "invalid part_rows shape {shape:?}"
                    )));
                }
                let rows = weights_tree.read_slice::<i32>("quantized.part_rows", shape).await?;
                if rows.is_empty()
                    || rows.iter().any(|&rows| rows <= 0)
                    || rows.iter().map(|&rows| i64::from(rows)).sum::<i64>() != i64::from(output_dimension)
                {
                    return Err(LinearBlockError::UnsupportedConfiguration(format!(
                        "part_rows must be positive and sum to {output_dimension}"
                    )));
                }
                let mut matrices = Vec::with_capacity(rows.len());
                for (index, rows) in rows.iter().copied().enumerate() {
                    let part_tree = weights_tree.subtree(&format!("quantized.parts.{index}"));
                    let spec = part_tree.metadata("spec")?;
                    matrices.push(load_matrix(rows as u32, spec, &part_tree).await?);
                }
                matrices.into_boxed_slice()
            },
            spec => {
                Box::new([load_matrix(output_dimension, spec, &weights_tree.subtree("quantized")).await?]) as Box<[_]>
            },
        };
        Ok(Self {
            transform: Transform::Kronecker {
                kernel,
                signs,
                orthogonal_factor,
                matrices,
                input_dimension,
                output_dimension,
            },
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
        let (hadamard, quantization_spec) = match spec {
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                incoherence_spec: Some(AnyIncoherenceProcessorSpec::HadamardSpec(hadamard)),
                quantization_spec,
                ..
            }) if hadamard.processing_mode == InputOutput => (hadamard, *quantization_spec),
            _ => {
                return Err(LinearBlockError::UnsupportedConfiguration(
                    "expected input-output Hadamard transform".into(),
                ));
            },
        };
        let (preparation, matrix) = Self::load_hadamard(
            input_dimension,
            output_dimension,
            data_type,
            data_type,
            data_type,
            hadamard,
            quantization_spec,
            false,
            &weights_tree,
            None,
            context,
        )
        .await?;
        Self::build_self_contained(data_type, preparation, matrix, context)
    }

    async fn load_hadamard(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        spec: HadamardSpec,
        quantization_spec: AnyWeightMatrixSpec,
        restore_output_basis: bool,
        weights_tree: &ParameterTree<'_, B>,
        bias_tree: Option<&ParameterTree<'_, B>>,
        context: &B::Context,
    ) -> Result<(LinearInputPreparation<B>, LinearMatmul<B>), LinearBlockError<B>> {
        if spec.block_size != HADAMARD_TRANSFORM_BLOCK_SIZE {
            return Err(LinearBlockError::UnsupportedConfiguration(format!("Hadamard block size {}", spec.block_size)));
        }
        let has_output_transform = spec.processing_mode == InputOutput;
        let rht_signs =
            weights_tree.read_buffer("incoherence_signs.input_signs", &[input_dimension], DataType::I32).await?;
        let output_factors = if has_output_transform && restore_output_basis {
            Some(weights_tree.read_buffer("incoherence_signs.output_signs", &[output_dimension], DataType::I32).await?)
        } else {
            if has_output_transform {
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
            transform: Transform::Hadamard {
                input_rht,
                matrix: inner_linear,
            },
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
        match &self.transform {
            Transform::Hadamard {
                input_rht,
                matrix,
            } => {
                let input = match input {
                    MatmulA::FullPrecision {
                        values,
                        offset,
                    } => {
                        let format = matrix.select_activation_format(batch_dim, None, command_buffer.context());
                        input_rht
                            .prepare_in_place(values, offset, batch_dim, format, command_buffer)
                            .map_err(LinearEncodeError::Backend)?
                    },
                    prepared => prepared,
                };
                matrix.encode_input(input, batch_dim, command_buffer)
            },
            Transform::Kronecker {
                kernel,
                signs,
                orthogonal_factor,
                matrices,
                input_dimension,
                output_dimension,
            } => {
                let input = match input {
                    MatmulA::FullPrecision {
                        values: input,
                        offset,
                    } => {
                        let mut values = command_buffer
                            .allocate_scratch_for_shape(&[batch_dim, *input_dimension], DataType::I8)
                            .map_err(LinearEncodeError::Backend)?;
                        let mut column_group_sums = command_buffer
                            .allocate_scratch_for_shape(&[batch_dim, COLUMN_GROUP_COUNT], DataType::F32)
                            .map_err(LinearEncodeError::Backend)?;
                        let mut scales = command_buffer
                            .allocate_scratch_for_shape(&[batch_dim], DataType::F32)
                            .map_err(LinearEncodeError::Backend)?;
                        kernel.encode(
                            (&input).subrange(offset..),
                            signs,
                            orthogonal_factor,
                            &mut values,
                            &mut column_group_sums,
                            &mut scales,
                            batch_dim,
                            command_buffer,
                        );
                        MatmulA::Trellis {
                            values,
                            column_group_sums,
                            scales,
                        }
                    },
                    prepared => prepared,
                };
                let mut output = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim, *output_dimension], DataType::BF16)
                    .map_err(LinearEncodeError::Backend)?;
                let mut column = 0;
                for matrix in matrices {
                    matrix.encode_columns(
                        input.as_ref(),
                        batch_dim,
                        &mut output,
                        column,
                        *output_dimension,
                        command_buffer,
                    )?;
                    column += matrix.output_dim();
                }
                Ok(output)
            },
        }
    }
}
