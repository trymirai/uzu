use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        gpu_types::{hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE, trellis::COLUMN_GROUP_COUNT},
        kernel::{TrellisTransformKernel, matmul::MatmulA, mixing_dimension},
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{
            AnyIncoherenceProcessorSpec, HadamardSpec,
            IncoherenceProcessingMode::{Input, InputOutput},
        },
    },
    data_type::DataType,
    encodable_block::linear::{
        Gather, Linear, LinearBlockError, LinearEncodeError, LinearInput, LinearMatmul, RHTLinearWrapper,
        input_rht::InputRht,
    },
    safetensors::{ParameterLoaderError, ParameterTree},
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
    rows: u32,
    tree: ParameterTree<'t, B>,
) -> Result<Vec<(u32, AnyWeightMatrixSpec, ParameterTree<'t, B>)>, ParameterLoaderError<B>> {
    if !matches!(spec, AnyWeightMatrixSpec::OutputStackSpec(_)) {
        return Ok(vec![(rows, spec, tree)]);
    }
    let shape = tree.shape("part_rows")?;
    let rows = tree.read_slice::<i32>("part_rows", shape).await?;
    let parts = tree.subtree("parts");
    rows.iter()
        .enumerate()
        .map(|(index, &rows)| {
            let tree = parts.subtree(&index.to_string());
            Ok((rows as u32, tree.metadata("spec")?, tree))
        })
        .collect()
}

impl<B: Backend> TransformedLinear<B> {
    pub(super) async fn new(
        input_dim: u32,
        output_dim: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        spec: AnyWeightMatrixSpec,
        weights_tree: ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, LinearBlockError<B>> {
        if has_biases {
            todo!("fuse biases into final output stores for output-stacked or Kronecker linears");
        }
        let parts = matrix_parts(spec, output_dim, weights_tree).await?;
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
                })) => {
                    let (preparation, linear) = RHTLinearWrapper::load_hadamard(
                        input_dim,
                        rows,
                        weights_data_type,
                        input_data_type,
                        output_data_type,
                        mode,
                        true,
                        quantization_spec,
                        &tree,
                        None,
                        context,
                    )
                    .await?;
                    let rht = InputRht::new(input_data_type, preparation, false, context)
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
        let MatmulA::FullPrecision {
            values: input,
            offset,
        } = input
        else {
            unreachable!();
        };
        let input = input.subrange(offset..);
        let command_buffer = &mut *command_buffer.debug_group("linear (transformed)");

        if self.groups.len() == 1 && self.groups[0].blocks.len() == 1 {
            let group = &self.groups[0];
            let block = &group.blocks[0];
            let a = self.prepare(group, input, batch_dim, command_buffer).map_err(LinearEncodeError::Backend)?;
            return block.encode_with_a(a.as_ref(), batch_dim, None::<Gather<&B::ScratchBuffer>>, command_buffer);
        }
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
