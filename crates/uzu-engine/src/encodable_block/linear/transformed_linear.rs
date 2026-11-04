use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        gpu_types::{hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE, trellis::COLUMN_GROUP_COUNT},
        kernel::{TrellisTransformKernel, matmul::MatmulA, mixing_dimension},
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{AnyIncoherenceProcessorSpec, HadamardSpec, HybridSpec, IncoherenceProcessingMode, KroneckerSpec},
    },
    data_type::DataType,
    encodable_block::linear::{
        Gather, Linear, LinearEncodeError, LinearInput, LinearMatmul, LinearMatmulError, RHTLinearWrapper,
        RHTLinearWrapperError, input_rht::InputRht,
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum TransformedLinearError<B: Backend> {
    #[error(transparent)]
    Hadamard(#[from] RHTLinearWrapperError<B>),
    #[error("Inner linear error: {0}")]
    InnerLinearError(#[from] LinearMatmulError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported transformed linear configuration: {0}")]
    UnsupportedConfiguration(String),
}

enum Rotation<B: Backend> {
    Kronecker {
        kernel: <B::Kernels as Kernels>::TrellisTransformKernel,
        signs: B::GlobalGpuBuffer,
        mixing: B::GlobalGpuBuffer,
    },
    Hadamard(InputRht<B>),
}

pub struct TransformedLinear<B: Backend> {
    groups: Box<[(Rotation<B>, Box<[LinearMatmul<B>]>)]>,
    input_dim: u32,
    output_dim: u32,
    output_data_type: DataType,
}

use TransformedLinearError::UnsupportedConfiguration as Unsupported;

async fn stack_parts<'t, B: Backend>(
    spec: AnyWeightMatrixSpec,
    rows: u32,
    tree: ParameterTree<'t, B>,
) -> Result<Vec<(u32, AnyWeightMatrixSpec, ParameterTree<'t, B>)>, TransformedLinearError<B>> {
    let AnyWeightMatrixSpec::OutputStackSpec(_) = spec else {
        return Ok(vec![(rows, spec, tree)]);
    };
    let part_rows_shape = tree.tensor_shape("part_rows")?;
    let [part_count] = part_rows_shape.as_ref() else {
        return Err(Unsupported(format!("invalid output-stack part_rows shape {part_rows_shape:?}")));
    };
    let part_rows = tree.read_slice::<i32>("part_rows", &part_rows_shape).await?;
    if part_rows.len() != *part_count as usize {
        return Err(Unsupported(format!("invalid output-stack part_rows shape {part_rows_shape:?}")));
    }
    let mut parts = Vec::with_capacity(part_rows.len());
    for (index, part_rows) in part_rows.into_iter().enumerate() {
        let part_rows = u32::try_from(part_rows)
            .ok()
            .filter(|part_rows| *part_rows > 0)
            .ok_or_else(|| Unsupported(format!("invalid output-stack part row count {part_rows}")))?;
        let part_tree = tree.subtree("parts").subtree(&index.to_string());
        let spec = part_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        parts.push((part_rows, spec, part_tree));
    }
    let stacked_rows = parts.iter().try_fold(0u32, |total, (part_rows, _, _)| total.checked_add(*part_rows));
    let Some(stacked_rows) = stacked_rows else {
        return Err(Unsupported("output-stack row count overflow".into()));
    };
    if stacked_rows != rows {
        return Err(Unsupported(format!("output-stack parts sum to {stacked_rows} rows, expected {rows}")));
    }
    Ok(parts)
}

impl<B: Backend> TransformedLinear<B> {
    pub(super) async fn new(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        spec: AnyWeightMatrixSpec,
        weights_tree: ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, TransformedLinearError<B>> {
        if has_biases {
            return Err(Unsupported("biases on output-stacked or Kronecker linears".into()));
        }
        let parts = stack_parts(spec, output_dimension, weights_tree).await?;
        let mut groups = Vec::with_capacity(parts.len());
        for (rows, spec, tree) in parts {
            let AnyWeightMatrixSpec::HybridSpec(hybrid) = spec else {
                return Err(Unsupported(format!("{spec:?} as an output-stack part")));
            };
            let signs_tree = tree.subtree("incoherence_signs");
            let quantized_tree = tree.subtree("quantized");
            groups.push(match hybrid {
                HybridSpec {
                    quantization_spec,
                    incoherence_spec: Some(AnyIncoherenceProcessorSpec::KroneckerSpec(KroneckerSpec {
                        processing_mode: IncoherenceProcessingMode::Input,
                        ..
                    })),
                    ..
                } => {
                    let data_types = [weights_data_type, input_data_type, output_data_type];
                    if data_types != [DataType::BF16; 3] {
                        return Err(Unsupported(format!("trellis linears are BF16 only, got {data_types:?}")));
                    }
                    let mixing_dimension = mixing_dimension(input_dimension);
                    // TrellisTransform (trellis.metal) handles 512..=4096-point Hadamards and at most 20480 columns.
                    let hadamard_size = input_dimension / mixing_dimension;
                    if !(512..=4096).contains(&hadamard_size) || input_dimension > 20480 {
                        return Err(Unsupported(format!("Kronecker rotation over {} columns", input_dimension)));
                    }
                    let kernel = <B::Kernels as Kernels>::TrellisTransformKernel::new(context, input_dimension)
                        .map_err(TransformedLinearError::BackendError)?;
                    let signs = signs_tree.read_buffer("signs", &[input_dimension], DataType::F32).await?;
                    let mixing =
                        signs_tree.read_buffer("small_q", &[mixing_dimension, mixing_dimension], DataType::F32).await?;
                    let rotation = Rotation::Kronecker {
                        kernel,
                        signs,
                        mixing,
                    };
                    let parts = stack_parts(*quantization_spec, rows, quantized_tree).await?;
                    let mut blocks = Vec::with_capacity(parts.len());
                    for (rows, spec, tree) in parts {
                        if !matches!(spec, AnyWeightMatrixSpec::QtipGaussianSpec(_)) {
                            return Err(Unsupported(format!("{spec:?} in a Kronecker linear")));
                        }
                        blocks.push(
                            LinearMatmul::load(
                                input_dimension,
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
                HybridSpec {
                    quantization_spec,
                    incoherence_spec: Some(AnyIncoherenceProcessorSpec::HadamardSpec(HadamardSpec {
                        block_size: HADAMARD_TRANSFORM_BLOCK_SIZE,
                        processing_mode:
                            mode @ (IncoherenceProcessingMode::Input | IncoherenceProcessingMode::InputOutput),
                        ..
                    })),
                    ..
                } => {
                    let (preparation, linear) = RHTLinearWrapper::load_hadamard(
                        input_dimension,
                        rows,
                        weights_data_type,
                        input_data_type,
                        output_data_type,
                        mode,
                        true,
                        *quantization_spec,
                        &tree,
                        None,
                        context,
                    )
                    .await?;
                    let rht = InputRht::new(input_data_type, preparation, false, context)
                        .map_err(TransformedLinearError::BackendError)?;
                    (Rotation::Hadamard(rht), Box::new([linear]) as Box<[_]>)
                },
                spec => return Err(Unsupported(format!("{spec:?} as an output-stack part"))),
            });
        }
        Ok(Self {
            groups: groups.into_boxed_slice(),
            input_dim: input_dimension,
            output_dim: output_dimension,
            output_data_type,
        })
    }

    fn prepare(
        &self,
        rotation: &Rotation<B>,
        input: impl BufferRef<Backend = B>,
        batch_dim: u32,
        linear: &LinearMatmul<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<LinearInput<B>, B::Error> {
        let (kernel, signs, mixing) = match rotation {
            Rotation::Hadamard(rht) => {
                let format = linear.select_activation_format(batch_dim, command_buffer.context());
                return rht.prepare(input, batch_dim, format, command_buffer);
            },
            Rotation::Kronecker {
                kernel,
                signs,
                mixing,
            } => (kernel, signs, mixing),
        };
        let mut values = command_buffer.allocate_scratch_for_shape(&[batch_dim, self.input_dim], DataType::I8)?;
        let mut column_group_sums =
            command_buffer.allocate_scratch_for_shape(&[batch_dim, COLUMN_GROUP_COUNT], DataType::F32)?;
        let mut scales = command_buffer.allocate_scratch_for_shape(&[batch_dim], DataType::F32)?;
        kernel.encode(
            input,
            signs,
            mixing,
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
        let MatmulA::FullPrecision {
            values: input,
            offset,
        } = input
        else {
            panic!("transformed linears take full-precision input");
        };
        let input = input.subrange(offset..);
        let command_buffer = &mut *command_buffer.debug_group("linear (transformed)");

        if let [(rotation, blocks)] = &*self.groups
            && let [block] = &**blocks
        {
            let a =
                self.prepare(rotation, input, batch_dim, block, command_buffer).map_err(LinearEncodeError::Backend)?;
            return block.encode_with_a(a.as_ref(), batch_dim, None::<Gather<&B::ScratchBuffer>>, command_buffer);
        }
        let mut output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.output_dim], self.output_data_type)
            .map_err(LinearEncodeError::Backend)?;
        let mut column = 0;
        for (rotation, blocks) in &self.groups {
            let a = self
                .prepare(rotation, input, batch_dim, &blocks[0], command_buffer)
                .map_err(LinearEncodeError::Backend)?;
            for block in blocks {
                block.encode_columns(a.as_ref(), batch_dim, &mut output, column, self.output_dim, command_buffer)?;
                column += block.output_dim;
            }
        }
        Ok(output)
    }
}
