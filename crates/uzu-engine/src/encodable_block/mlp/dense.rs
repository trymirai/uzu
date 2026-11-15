//! MLP block encodable.

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::{
            JangtqDenseDownKernel, JangtqDenseDownMmaKernel, JangtqDenseFusedGateUpKernel,
            JangtqDenseFusedGateUpMmaKernel, JangtqHadamard32Kernel, Kernels,
            matmul::{ActivationFormat, MatmulA},
        },
    },
    config::{mlp::dense_mlp::DenseMLPConfig, weight_matrix::AnyWeightMatrixSpec},
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError, LinearInput, LinearInputPreparation},
        mlp::{Mlp, MlpEncodeError, gate_act_mul::MlpGateActMulEncodable},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

const JANGTQ_PREFILL_MMA_MIN_BATCH: u32 = 8;

#[derive(Debug)]
struct DenseJangtqPath<B: Backend> {
    model_dimension: u32,
    hidden_dimension: u32,
    data_type: DataType,
    words_gu: u32,
    words_d: u32,
    alpha_g: f32,
    beta_g: f32,
    alpha_u: f32,
    beta_u: f32,
    alpha_d: f32,
    beta_d: f32,
    packed_g: B::GlobalGpuBuffer,
    scales_g: B::GlobalGpuBuffer,
    packed_u: B::GlobalGpuBuffer,
    scales_u: B::GlobalGpuBuffer,
    packed_d: B::GlobalGpuBuffer,
    scales_d: B::GlobalGpuBuffer,
    hadamard32: <B::Kernels as Kernels>::JangtqHadamard32Kernel,
    gate_up_decode: <B::Kernels as Kernels>::JangtqDenseFusedGateUpKernel,
    gate_up_mma: <B::Kernels as Kernels>::JangtqDenseFusedGateUpMmaKernel,
    down_decode: <B::Kernels as Kernels>::JangtqDenseDownKernel,
    down_mma: <B::Kernels as Kernels>::JangtqDenseDownMmaKernel,
}

#[derive(Debug)]
enum DenseMlpImpl<B: Backend> {
    Standard {
        up: Box<dyn Linear<B>>,
        gate: MlpGateActMulEncodable<B>,
        down: Box<dyn Linear<B>>,
    },
    Jangtq(Box<DenseJangtqPath<B>>),
}

pub struct DenseMlp<B: Backend> {
    inner: DenseMlpImpl<B>,
}

#[derive(Debug, Error)]
pub enum DenseMlpNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
}

impl<B: Backend> DenseMlp<B> {
    async fn try_load_jangtq(
        model_dimension: u32,
        hidden_dimension: u32,
        data_type: DataType,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Option<Self>, DenseMlpNewError<B>> {
        let gate_weights = parameter_tree.subtree("gate_projection").subtree("weights");
        let gate_spec = match gate_weights.metadata::<AnyWeightMatrixSpec>("spec") {
            Ok(AnyWeightMatrixSpec::Jangtq2Spec(spec)) => spec,
            Ok(_) | Err(ParameterLoaderError::KeyNotFound(_)) => return Ok(None),
            Err(err) => return Err(LinearBlockError::ParameterError(err).into()),
        };

        let up_weights = parameter_tree.subtree("up_projection").subtree("weights");
        let AnyWeightMatrixSpec::Jangtq2Spec(up_spec) =
            up_weights.metadata::<AnyWeightMatrixSpec>("spec").map_err(LinearBlockError::ParameterError)?
        else {
            return Err(LinearBlockError::UnsupportedConfiguration(
                "Expected Jangtq2Spec on dense MLP up_projection".into(),
            )
            .into());
        };
        let down_weights = parameter_tree.subtree("down_projection").subtree("weights");
        let AnyWeightMatrixSpec::Jangtq2Spec(down_spec) =
            down_weights.metadata::<AnyWeightMatrixSpec>("spec").map_err(LinearBlockError::ParameterError)?
        else {
            return Err(LinearBlockError::UnsupportedConfiguration(
                "Expected Jangtq2Spec on dense MLP down_projection".into(),
            )
            .into());
        };

        if gate_spec.bits != 3 || up_spec.bits != 3 || down_spec.bits != 3 {
            return Err(LinearBlockError::UnsupportedConfiguration(format!(
                "Dense JANGTQ path requires 3-bit weights, got gate={} up={} down={}",
                gate_spec.bits, up_spec.bits, down_spec.bits
            ))
            .into());
        }
        if model_dimension % 512 != 0 || hidden_dimension % 512 != 0 {
            return Err(LinearBlockError::UnsupportedConfiguration(format!(
                "Dense JANGTQ requires dims divisible by 512, got K={model_dimension} H={hidden_dimension}"
            ))
            .into());
        }

        let words_gu = (model_dimension * 3) / 32;
        let words_d = (hidden_dimension * 3) / 32;

        let gate_q = gate_weights.subtree("quantized");
        let up_q = up_weights.subtree("quantized");
        let down_q = down_weights.subtree("quantized");

        let packed_g =
            gate_q.read_buffer("weights", &[hidden_dimension, words_gu], DataType::U32).await.map_err(LinearBlockError::ParameterError)?;
        let scales_g =
            gate_q.read_buffer("scales", &[hidden_dimension], DataType::F16).await.map_err(LinearBlockError::ParameterError)?;
        let packed_u =
            up_q.read_buffer("weights", &[hidden_dimension, words_gu], DataType::U32).await.map_err(LinearBlockError::ParameterError)?;
        let scales_u =
            up_q.read_buffer("scales", &[hidden_dimension], DataType::F16).await.map_err(LinearBlockError::ParameterError)?;
        let packed_d =
            down_q.read_buffer("weights", &[model_dimension, words_d], DataType::U32).await.map_err(LinearBlockError::ParameterError)?;
        let scales_d =
            down_q.read_buffer("scales", &[model_dimension], DataType::F16).await.map_err(LinearBlockError::ParameterError)?;

        let hadamard32 = <B::Kernels as Kernels>::JangtqHadamard32Kernel::new(context).map_err(DenseMlpNewError::Backend)?;
        let gate_up_decode =
            <B::Kernels as Kernels>::JangtqDenseFusedGateUpKernel::new(context).map_err(DenseMlpNewError::Backend)?;
        let gate_up_mma =
            <B::Kernels as Kernels>::JangtqDenseFusedGateUpMmaKernel::new(context).map_err(DenseMlpNewError::Backend)?;
        let down_decode =
            <B::Kernels as Kernels>::JangtqDenseDownKernel::new(context, data_type).map_err(DenseMlpNewError::Backend)?;
        let down_mma =
            <B::Kernels as Kernels>::JangtqDenseDownMmaKernel::new(context, data_type).map_err(DenseMlpNewError::Backend)?;

        Ok(Some(Self {
            inner: DenseMlpImpl::Jangtq(Box::new(DenseJangtqPath {
                model_dimension,
                hidden_dimension,
                data_type,
                words_gu,
                words_d,
                alpha_g: gate_spec.alpha,
                beta_g: gate_spec.beta,
                alpha_u: up_spec.alpha,
                beta_u: up_spec.beta,
                alpha_d: down_spec.alpha,
                beta_d: down_spec.beta,
                packed_g,
                scales_g,
                packed_u,
                scales_u,
                packed_d,
                scales_d,
                hadamard32,
                gate_up_decode,
                gate_up_mma,
                down_decode,
                down_mma,
            })),
        }))
    }

    pub async fn new(
        model_dimension: u32,
        hidden_dimension: u32,
        data_type: DataType,
        config: &DenseMLPConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Self, Option<B::GlobalGpuBuffer>), DenseMlpNewError<B>> {
        if let Some(jangtq) =
            Self::try_load_jangtq(model_dimension, hidden_dimension, data_type, parameter_tree, context).await?
        {
            return Ok((jangtq, None));
        }
        let (up, up_input_hadamard_factors) = <dyn Linear<B>>::new_with_input_rht(
            model_dimension,
            2 * hidden_dimension,
            data_type,
            config.has_up_biases,
            &parameter_tree.subtree("up_projection"),
            context,
        )
        .await?;
        let mlp =
            Self::load_gate_and_down(model_dimension, hidden_dimension, data_type, up, config, parameter_tree, context)
                .await?;

        Ok((mlp, up_input_hadamard_factors))
    }

    pub async fn new_for_fused_input(
        model_dimension: u32,
        hidden_dimension: u32,
        data_type: DataType,
        config: &DenseMLPConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Self, Option<LinearInputPreparation<B>>), DenseMlpNewError<B>> {
        if let Some(jangtq) =
            Self::try_load_jangtq(model_dimension, hidden_dimension, data_type, parameter_tree, context).await?
        {
            return Ok((jangtq, None));
        }
        let (up, up_input_preparation) = <dyn Linear<B>>::new_for_fused_input(
            model_dimension,
            2 * hidden_dimension,
            data_type,
            config.has_up_biases,
            &parameter_tree.subtree("up_projection"),
            context,
        )
        .await?;
        let mlp =
            Self::load_gate_and_down(model_dimension, hidden_dimension, data_type, up, config, parameter_tree, context)
                .await?;

        Ok((mlp, up_input_preparation))
    }

    pub fn encode_input(
        &self,
        input: LinearInput<B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MlpEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("mlp (dense)");

        match &self.inner {
            DenseMlpImpl::Standard { up, gate, down } => {
                let fused_up = up.encode_input(input, batch_dim, command_buffer)?;
                let act_format = down.select_activation_format(batch_dim, None, command_buffer.context());
                let down_input =
                    gate.encode_for_linear(&fused_up, batch_dim, act_format, command_buffer).map_err(MlpEncodeError::Backend)?;
                let output = down.encode_input(down_input, batch_dim, command_buffer)?;
                Ok(output)
            },
            DenseMlpImpl::Jangtq(path) => {
                let mut output = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim, path.model_dimension], path.data_type)
                    .map_err(MlpEncodeError::Backend)?;
                if batch_dim == 0 {
                    return Ok(output);
                }
                let MatmulA::FullPrecision { values, offset } = input else {
                    unreachable!("Dense JANGTQ MLP only supports FullPrecision inputs");
                };
                let mut rotated_x = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim, path.model_dimension], DataType::F32)
                    .map_err(MlpEncodeError::Backend)?;
                let mut hidden = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim, path.hidden_dimension], DataType::F32)
                    .map_err(MlpEncodeError::Backend)?;

                let x_sub = values.subrange(offset..);
                path.hadamard32.encode(x_sub, &mut rotated_x, path.model_dimension, batch_dim, command_buffer);

                if batch_dim < JANGTQ_PREFILL_MMA_MIN_BATCH {
                    path.gate_up_decode.encode(
                        &rotated_x,
                        &path.packed_g,
                        &path.scales_g,
                        &path.packed_u,
                        &path.scales_u,
                        &mut hidden,
                        path.alpha_g,
                        path.beta_g,
                        path.alpha_u,
                        path.beta_u,
                        path.model_dimension,
                        path.hidden_dimension,
                        path.words_gu,
                        batch_dim,
                        command_buffer,
                    );
                    path.down_decode.encode(
                        &hidden,
                        &path.packed_d,
                        &path.scales_d,
                        &mut output,
                        path.alpha_d,
                        path.beta_d,
                        path.hidden_dimension,
                        path.model_dimension,
                        path.words_d,
                        batch_dim,
                        command_buffer,
                    );
                } else {
                    path.gate_up_mma.encode(
                        &rotated_x,
                        &path.packed_g,
                        &path.scales_g,
                        &path.packed_u,
                        &path.scales_u,
                        &mut hidden,
                        path.alpha_g,
                        path.beta_g,
                        path.alpha_u,
                        path.beta_u,
                        path.model_dimension,
                        path.hidden_dimension,
                        path.words_gu,
                        batch_dim,
                        command_buffer,
                    );
                    path.down_mma.encode(
                        &hidden,
                        &path.packed_d,
                        &path.scales_d,
                        &mut output,
                        path.alpha_d,
                        path.beta_d,
                        path.hidden_dimension,
                        path.model_dimension,
                        path.words_d,
                        batch_dim,
                        command_buffer,
                    );
                }

                Ok(output)
            },
        }
    }

    pub fn select_activation_format(
        &self,
        batch_dim: u32,
        output_dim: Option<u32>,
        context: &B::Context,
    ) -> ActivationFormat {
        match &self.inner {
            DenseMlpImpl::Standard { up, .. } => up.select_activation_format(batch_dim, output_dim, context),
            DenseMlpImpl::Jangtq(_) => ActivationFormat::Bf16,
        }
    }

    async fn load_gate_and_down(
        model_dimension: u32,
        hidden_dimension: u32,
        data_type: DataType,
        up: Box<dyn Linear<B>>,
        config: &DenseMLPConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, DenseMlpNewError<B>> {
        let (down, down_input_preparation) = <dyn Linear<B>>::new_for_fused_input(
            hidden_dimension,
            model_dimension,
            data_type,
            config.has_down_biases,
            &parameter_tree.subtree("down_projection"),
            context,
        )
        .await?;

        let gate = MlpGateActMulEncodable::new(
            hidden_dimension,
            data_type,
            config.activation.clone(),
            config.gate_clipping,
            config.up_clipping,
            down_input_preparation,
            context,
        )
        .map_err(DenseMlpNewError::Backend)?;

        Ok(Self {
            inner: DenseMlpImpl::Standard {
                up,
                gate,
                down,
            },
        })
    }
}

impl<B: Backend> Mlp<B> for DenseMlp<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MlpEncodeError<B>> {
        self.encode_input(MatmulA::full_precision(input), batch_dim, command_buffer)
    }
}
