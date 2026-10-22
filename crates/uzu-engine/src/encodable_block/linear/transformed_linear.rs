use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding, Kernels,
        gpu_types::{hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE, trellis::COLUMN_GROUP_COUNT},
        kernel::{TrellisTransformKernel, mixing_dimension},
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{HybridSpec, IncoherenceKind, IncoherenceProcessingMode},
    },
    data_type::DataType,
    encodable_block::linear::{
        Gather, Linear, LinearEncodeError, LinearInput, LinearInputPreparation, LinearMatmul, LinearMatmulError,
        input_rht::InputRht,
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum TransformedLinearError<B: Backend> {
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
        signs: B::GlobalBuffer,
        mixing: B::GlobalBuffer,
    },
    Hadamard(InputRht<B>),
}

pub struct TransformedLinear<B: Backend> {
    groups: Box<[(Rotation<B>, Box<[LinearMatmul<B>]>)]>,
    input_dim: u32,
    output_dim: u32,
    output_data_type: DataType,
}

struct Shape {
    input_dim: u32,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
}

fn unsupported<B: Backend>(message: String) -> TransformedLinearError<B> {
    TransformedLinearError::UnsupportedConfiguration(message)
}

fn row_stack_parts<'t, B: Backend>(
    spec: AnyWeightMatrixSpec,
    rows: u32,
    tree: ParameterTree<'t, B>,
) -> Result<Vec<(u32, AnyWeightMatrixSpec, ParameterTree<'t, B>)>, TransformedLinearError<B>> {
    let AnyWeightMatrixSpec::RowStackSpec(stack) = spec else {
        return Ok(vec![(rows, spec, tree)]);
    };
    let stacked_rows: u32 = stack.parts.iter().map(|(rows, _)| rows).sum();
    if stacked_rows != rows {
        return Err(unsupported(format!("row-stack parts sum to {stacked_rows} rows, expected {rows}")));
    }
    Ok(stack
        .parts
        .into_iter()
        .enumerate()
        .map(|(index, (rows, spec))| (rows, spec, tree.subtree("parts").subtree(&index.to_string())))
        .collect())
}

impl<B: Backend> TransformedLinear<B> {
    pub(super) fn new(
        input_dimension: u32,
        output_dimension: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        has_biases: bool,
        spec: AnyWeightMatrixSpec,
        weights_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, TransformedLinearError<B>> {
        if has_biases {
            return Err(unsupported("biases on row-stacked or Kronecker linears".into()));
        }
        let shape = Shape {
            input_dim: input_dimension,
            weights_data_type,
            input_data_type,
            output_data_type,
        };
        let groups = row_stack_parts(spec, output_dimension, weights_tree.clone())?
            .into_iter()
            .map(|(rows, spec, tree)| match spec {
                AnyWeightMatrixSpec::HybridSpec(hybrid) => Self::load_group(&shape, rows, hybrid, &tree, context),
                spec => Err(unsupported(format!("{spec:?} as a row-stack part"))),
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            groups,
            input_dim: input_dimension,
            output_dim: output_dimension,
            output_data_type,
        })
    }

    fn load_group(
        shape: &Shape,
        rows: u32,
        spec: HybridSpec,
        tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Rotation<B>, Box<[LinearMatmul<B>]>), TransformedLinearError<B>> {
        let signs_tree = tree.subtree("incoherence_signs");
        let quantized_tree = tree.subtree("quantized");
        let load = |rows, spec, tree: &ParameterTree<B>, output_factors| {
            LinearMatmul::load(
                shape.input_dim,
                rows,
                shape.weights_data_type,
                shape.input_data_type,
                shape.output_data_type,
                spec,
                output_factors,
                tree,
                None,
                context,
            )
        };
        match spec {
            HybridSpec {
                quantization_spec,
                incoherence_processing_mode: IncoherenceProcessingMode::Input,
                incoherence_kind: IncoherenceKind::Kronecker,
                ..
            } => {
                let data_types = [shape.weights_data_type, shape.input_data_type, shape.output_data_type];
                if data_types != [DataType::BF16; 3] {
                    return Err(unsupported(format!("trellis linears are BF16 only, got {data_types:?}")));
                }
                let mixing_dimension = mixing_dimension(shape.input_dim);
                // TrellisTransform (trellis.metal) handles 512..=4096-point Hadamards and at most 20480 columns.
                let hadamard_size = shape.input_dim / mixing_dimension;
                if !(512..=4096).contains(&hadamard_size) || shape.input_dim > 20480 {
                    return Err(unsupported(format!("Kronecker rotation over {} columns", shape.input_dim)));
                }
                let rotation = Rotation::Kronecker {
                    kernel: <B::Kernels as Kernels>::TrellisTransformKernel::new(context, shape.input_dim)
                        .map_err(TransformedLinearError::BackendError)?,
                    signs: signs_tree.read_buffer("signs", &[shape.input_dim], DataType::F32)?,
                    mixing: signs_tree.read_buffer("small_q", &[mixing_dimension, mixing_dimension], DataType::F32)?,
                };
                let blocks = row_stack_parts(*quantization_spec, rows, quantized_tree)?
                    .into_iter()
                    .map(|(rows, spec, tree)| match spec {
                        spec @ AnyWeightMatrixSpec::QtipGaussianSpec(_) => Ok(load(rows, spec, &tree, None)?),
                        spec => Err(unsupported(format!("{spec:?} in a Kronecker linear"))),
                    })
                    .collect::<Result<_, _>>()?;
                Ok((rotation, blocks))
            },
            HybridSpec {
                quantization_spec,
                incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
                incoherence_processing_mode:
                    mode @ (IncoherenceProcessingMode::Input | IncoherenceProcessingMode::InputOutput),
                incoherence_kind: IncoherenceKind::Hadamard,
                ..
            } => {
                let rht_signs = signs_tree.read_buffer("input_signs", &[shape.input_dim], DataType::I32)?;
                let output_factors = (mode == IncoherenceProcessingMode::InputOutput)
                    .then(|| signs_tree.read_buffer("output_signs", &[rows], DataType::I32))
                    .transpose()?;
                let mut linear = load(rows, *quantization_spec, &quantized_tree, output_factors)?;
                let activation_quantization = linear.prepare_a8(context);
                if activation_quantization.is_some() {
                    linear.optimize_weights_layout(context).map_err(TransformedLinearError::BackendError)?;
                }
                let preparation = LinearInputPreparation {
                    rht_signs,
                    activation_quantization,
                };
                let rht = InputRht::new(shape.input_data_type, preparation, false, context)
                    .map_err(TransformedLinearError::BackendError)?;
                Ok((Rotation::Hadamard(rht), Box::new([linear])))
            },
            spec => Err(unsupported(format!("{spec:?} as a row-stack part"))),
        }
    }

    fn prepare(
        &self,
        rotation: &Rotation<B>,
        input: &B::ScratchBuffer,
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
        Ok(LinearInput::Trellis {
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
        let LinearInput::FullPrecision(input) = input else {
            panic!("transformed linears take full-precision input");
        };
        let command_buffer = &mut *command_buffer.debug_group("linear (transformed)");

        if let [(rotation, blocks)] = &*self.groups
            && let [block] = &**blocks
        {
            let a =
                self.prepare(rotation, &input, batch_dim, block, command_buffer).map_err(LinearEncodeError::Backend)?;
            return block.encode_with_a(a.as_matmul_a(), batch_dim, None::<Gather<&B::ScratchBuffer>>, command_buffer);
        }
        let mut output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.output_dim], self.output_data_type)
            .map_err(LinearEncodeError::Backend)?;
        let mut column = 0;
        for (rotation, blocks) in &self.groups {
            let a = self
                .prepare(rotation, &input, batch_dim, &blocks[0], command_buffer)
                .map_err(LinearEncodeError::Backend)?;
            for block in blocks {
                block.encode_columns(
                    a.as_matmul_a(),
                    batch_dim,
                    &mut output,
                    column,
                    self.output_dim,
                    command_buffer,
                )?;
                column += block.output_dim;
            }
        }
        Ok(output)
    }
}
