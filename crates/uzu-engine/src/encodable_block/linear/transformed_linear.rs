use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
        gpu_types::{HADAMARD_TRANSFORM_BLOCK_SIZE, trellis::mixing_dimension},
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{HybridSpec, IncoherenceKind, IncoherenceProcessingMode},
    },
    data_type::DataType,
    encodable_block::linear::{
        Gather, Linear, LinearInput, LinearInputPreparation, LinearMatmul, LinearMatmulError,
        input_transform::{InputRht, InputTransform, TrellisRotation},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum TransformedLinearError<B: Backend> {
    #[error("Inner linear error: {0}")]
    InnerLinearError(#[from] LinearMatmulError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported RHT linear configuration: {0}")]
    UnsupportedConfiguration(String),
}

pub struct TransformedLinear<B: Backend> {
    groups: Box<[(InputTransform<B>, Box<[LinearMatmul<B>]>)]>,
}

enum InputRotation<B: Backend> {
    Rht(LinearInputPreparation<B>),
    Trellis(TrellisRotation<B>),
}

impl<B: Backend> TransformedLinear<B> {
    pub(super) fn try_new_with_input_preparation(
        context: &B::Context,
        input_dimension: u32,
        output_dimension: u32,
        has_biases: bool,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        allow_prequantized_activation: bool,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Option<(Box<dyn Linear<B>>, Option<LinearInputPreparation<B>>)>, TransformedLinearError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        if !matches!(
            spec,
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                adapter_spec: None,
                incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
                incoherence_processing_mode: IncoherenceProcessingMode::InputOutput,
                incoherence_kind: IncoherenceKind::Hadamard,
                ..
            })
        ) {
            return Ok(None);
        }

        let mut parts = Self::load(
            context,
            spec,
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            weights_tree,
            has_biases.then_some(parameter_tree),
        )?;
        let (InputRotation::Rht(preparation), blocks) = parts.remove(0) else {
            unreachable!("Hadamard specs load an RHT rotation")
        };
        if preparation.activation_quantization.is_some() && !allow_prequantized_activation {
            let wrapper = Self::from_groups(context, input_data_type, vec![(InputRotation::Rht(preparation), blocks)])?;
            Ok(Some((Box::new(wrapper), None)))
        } else {
            Ok(Some((Box::new(blocks.into_vec().remove(0)), Some(preparation))))
        }
    }

    pub(super) fn new_stacked(
        context: &B::Context,
        spec: AnyWeightMatrixSpec,
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        weights_tree: ParameterTree<B>,
        biases: Option<&ParameterTree<B>>,
    ) -> Result<Self, TransformedLinearError<B>> {
        let parts = Self::load(
            context,
            spec,
            input_dimension,
            output_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
            weights_tree,
            biases,
        )?;
        Self::from_groups(context, input_data_type, parts)
    }

    fn from_groups(
        context: &B::Context,
        input_data_type: DataType,
        parts: Vec<(InputRotation<B>, Box<[LinearMatmul<B>]>)>,
    ) -> Result<Self, TransformedLinearError<B>> {
        let in_place = parts.len() == 1;
        let groups = parts
            .into_iter()
            .map(|(rotation, blocks)| {
                let transform = match rotation {
                    InputRotation::Rht(preparation) => InputTransform::Rht(
                        InputRht::new(context, input_data_type, preparation, in_place)
                            .map_err(TransformedLinearError::BackendError)?,
                    ),
                    InputRotation::Trellis(rotation) => InputTransform::Trellis(rotation),
                };
                Ok((transform, blocks))
            })
            .collect::<Result<_, TransformedLinearError<B>>>()?;
        Ok(Self {
            groups,
        })
    }

    fn load(
        context: &B::Context,
        spec: AnyWeightMatrixSpec,
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        weights_tree: ParameterTree<B>,
        biases: Option<&ParameterTree<B>>,
    ) -> Result<Vec<(InputRotation<B>, Box<[LinearMatmul<B>]>)>, TransformedLinearError<B>> {
        let load = |spec, rows, tree: &ParameterTree<B>, bias_tree, output_factors| {
            LinearMatmul::load(
                context,
                spec,
                input_dimension,
                rows,
                weights_data_type,
                input_data_type,
                output_data_type,
                tree,
                bias_tree,
                output_factors,
            )
        };
        let parts = Self::row_stack_parts(spec, output_dimension, weights_tree);
        let biases = biases.filter(|_| parts.len() == 1);
        parts
            .into_iter()
            .map(|(rows, spec, tree)| {
                let AnyWeightMatrixSpec::HybridSpec(hybrid) = spec else {
                    unreachable!("row-stack parts are hybrid specs")
                };
                let signs_tree = tree.subtree("incoherence_signs");
                let quantized_tree = tree.subtree("quantized");
                if hybrid.incoherence_kind == IncoherenceKind::Kronecker {
                    let mixing_dimension = mixing_dimension(input_dimension);
                    let signs = signs_tree.leaf("signs")?.validate(&[input_dimension], DataType::F32)?.read_buffer()?;
                    let mixing = signs_tree
                        .leaf("small_q")?
                        .validate(&[mixing_dimension, mixing_dimension], DataType::F32)?
                        .read_buffer()?;
                    let rotation = TrellisRotation::new(context, signs, mixing, input_dimension)
                        .map_err(TransformedLinearError::BackendError)?;
                    let blocks = Self::row_stack_parts(*hybrid.quantization_spec, rows, quantized_tree)
                        .into_iter()
                        .map(|(rows, spec, tree)| load(spec, rows, &tree, None, None))
                        .collect::<Result<_, _>>()?;
                    return Ok((InputRotation::Trellis(rotation), blocks));
                }
                let rht_signs =
                    signs_tree.leaf("input_signs")?.validate(&[input_dimension], DataType::I32)?.read_buffer()?;
                let output_factors = (hybrid.incoherence_processing_mode == IncoherenceProcessingMode::InputOutput)
                    .then(|| signs_tree.leaf("output_signs")?.validate(&[rows], DataType::I32)?.read_buffer())
                    .transpose()?;
                let mut linear = load(*hybrid.quantization_spec, rows, &quantized_tree, biases, output_factors)?;
                let activation_quantization = linear.prepare_a8(context);
                let preparation = LinearInputPreparation {
                    rht_signs,
                    activation_quantization,
                };
                Ok((InputRotation::Rht(preparation), Box::new([linear]) as Box<[_]>))
            })
            .collect()
    }

    fn row_stack_parts<'t>(
        spec: AnyWeightMatrixSpec,
        rows: u32,
        tree: ParameterTree<'t, B>,
    ) -> Vec<(u32, AnyWeightMatrixSpec, ParameterTree<'t, B>)> {
        match spec {
            AnyWeightMatrixSpec::RowStackSpec(stack) => stack
                .parts
                .into_iter()
                .enumerate()
                .map(|(index, (rows, spec))| (rows, spec, tree.subtree("parts").subtree(&index.to_string())))
                .collect(),
            spec => vec![(rows, spec, tree)],
        }
    }

    fn encode_blocks(
        blocks: &[LinearMatmul<B>],
        a: &LinearInput<B>,
        batch_dim: u32,
        output_dim: u32,
        column: &mut u32,
        output: &mut B::ScratchBuffer,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        for block in blocks {
            let row_stride = (block.output_dim != output_dim).then_some(output_dim);
            let gather = None::<Gather<&B::ScratchBuffer>>;
            if row_stride.is_none() || block.is_trellis() {
                block.encode_into(a.as_matmul_a(), batch_dim, gather, output, *column, row_stride, command_buffer)?;
            } else {
                // Only trellis GEMM writes strided: other blocks run contiguous, then copy into their columns.
                let part = block.encode_with_a(a.as_matmul_a(), batch_dim, gather, command_buffer)?;
                let element = block.output_data_type.size_in_bytes();
                let row_bytes = block.output_dim as usize * element;
                for row in 0..batch_dim as usize {
                    let start = (row * output_dim as usize + *column as usize) * element;
                    command_buffer.encode_copy(
                        part.subrange(row * row_bytes..(row + 1) * row_bytes),
                        output.subrange_mut(start..start + row_bytes),
                    );
                }
            }
            *column += block.output_dim;
        }
        Ok(())
    }
}

impl<B: Backend> Linear<B> for TransformedLinear<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        command_buffer.push_debug_group("linear (transformed)");
        if let [(transform, blocks)] = &*self.groups
            && let [block] = &**blocks
        {
            let a = transform.prepare_in_place(input, batch_dim, block, command_buffer)?;
            let output =
                block.encode_with_a(a.as_matmul_a(), batch_dim, None::<Gather<&B::ScratchBuffer>>, command_buffer);
            command_buffer.pop_debug_group();
            return output;
        }

        let output_dim = self.groups.iter().flat_map(|(_, blocks)| blocks).map(|block| block.output_dim).sum();
        let mut output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, output_dim], self.groups[0].1[0].output_data_type)?;
        if let [(transform, blocks)] = &*self.groups {
            let a = transform.prepare_in_place(input, batch_dim, &blocks[0], command_buffer)?;
            Self::encode_blocks(blocks, &a, batch_dim, output_dim, &mut 0, &mut output, command_buffer)?;
        } else {
            let mut column = 0;
            for (transform, blocks) in &self.groups {
                let a = transform.prepare(&input, batch_dim, &blocks[0], command_buffer)?;
                Self::encode_blocks(blocks, &a, batch_dim, output_dim, &mut column, &mut output, command_buffer)?;
            }
        }

        command_buffer.pop_debug_group();
        Ok(output)
    }
}
