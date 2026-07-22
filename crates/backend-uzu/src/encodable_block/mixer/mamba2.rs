use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder, Kernels,
        gpu_types::ActivationType,
        kernel::{
            Conv1dDecodeKernel, Conv1dPackKernel, Conv1dScanKernel, SSDPrefill64Kernel, SSDPrefillKernel,
            SSDUpdateKernel, SplitInProjKernel,
        },
    },
    config::token_mixer::mamba2::Mamba2Config,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError},
        mixer::{Mixer, MixerState, attention::rope::PrecalculatedRoPE},
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

pub struct Mamba2State<B: Backend> {
    conv_state: Allocation<B>,
    ssm_state: Allocation<B>,
    suffix_length: Option<usize>,
}

impl<B: Backend> MixerState<B> for Mamba2State<B> {
    fn prepare(
        &mut self,
        _context_length: usize,
        _suffix_length: usize,
        _context: &B::Context,
    ) -> Result<(), B::Error> {
        Ok(())
    }

    fn encode_accept(
        &mut self,
        accepted_indices: &[usize],
        _encoder: &mut Encoder<B>,
    ) -> Result<(), <B as Backend>::Error> {
        assert!(self.suffix_length.take() == Some(*accepted_indices.last().unwrap() + 1));
        Ok(())
    }
}

enum Mamba2SSDPrefillVariant<B: Backend> {
    Universal(<B::Kernels as Kernels>::SSDPrefillKernel),
    Special64(<B::Kernels as Kernels>::SSDPrefill64Kernel),
}

pub struct Mamba2<B: Backend> {
    kernel_size: usize,
    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
    state_dim: usize,
    inner_dim: usize,
    conv_dim: usize,
    inner_data_type: DataType,
    activation_type: ActivationType,
    in_projection: Box<dyn Linear<B>>,
    gate_bias: Allocation<B>,
    split_inproj: <B::Kernels as Kernels>::SplitInProjKernel,
    conv_weight: Allocation<B>,
    conv_bias: Option<Allocation<B>>,
    conv_decode: <B::Kernels as Kernels>::Conv1dDecodeKernel,
    conv_pack: <B::Kernels as Kernels>::Conv1dPackKernel,
    conv_scan: <B::Kernels as Kernels>::Conv1dScanKernel,
    skip_connection_weight: Allocation<B>,
    ssd_update: <B::Kernels as Kernels>::SSDUpdateKernel,
    ssd_prefill: Mamba2SSDPrefillVariant<B>,
    out_projection: Box<dyn Linear<B>>,
}

#[derive(Debug, Error)]
pub enum Mamba2NewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
}

impl<B: Backend> Mamba2<B> {
    pub fn new(
        hidden_dim: usize,
        outer_data_type: DataType,
        config: &Mamba2Config,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Self, Option<Allocation<B>>), Mamba2NewError<B>> {
        let inner_data_type = DataType::F32;

        let kernel_size = config.kernel_size;
        let num_heads = config.num_heads;
        let num_groups = config.num_groups;
        let head_dim = config.head_dim;
        let state_dim = config.state_dim;

        let inner_dim = num_heads * head_dim;
        let conv_dim = inner_dim + 2 * num_groups * state_dim;

        let activation_type = config.activation.act_type();

        if kernel_size <= 1 {
            return Err(Mamba2NewError::UnsupportedConfiguration(format!("kernel_size = {kernel_size} (must be > 1)")));
        }

        let (in_projection, in_projection_input_hadamard_factors) =
            <dyn Linear<B>>::new_extracting_input_hadamard_mixed_precision(
                hidden_dim,
                [conv_dim, inner_dim, num_heads],
                config.has_in_biases,
                context,
                outer_data_type,
                outer_data_type,
                inner_data_type,
                &parameter_tree.subtree("in_projection")?,
            )?;

        let gate_bias = parameter_tree.leaf("gate_bias")?.validate(&[inner_dim], inner_data_type)?.read_allocation()?;
        let split_inproj = <B::Kernels as Kernels>::SplitInProjKernel::new(context, inner_data_type)
            .map_err(Mamba2NewError::Backend)?;

        let conv_config = &config.conv_config;
        let conv_tree = parameter_tree.subtree("conv")?;

        let conv_weight =
            conv_tree.leaf("weights")?.validate(&[conv_dim, kernel_size], inner_data_type)?.read_allocation()?;
        let conv_bias = if conv_config.has_biases {
            Some(conv_tree.leaf("biases")?.validate(&[conv_dim], inner_data_type)?.read_allocation()?)
        } else {
            None
        };
        let conv_decode =
            <B::Kernels as Kernels>::Conv1dDecodeKernel::new(context, inner_data_type, conv_config.has_biases, true)
                .map_err(Mamba2NewError::Backend)?;
        let conv_pack = <B::Kernels as Kernels>::Conv1dPackKernel::new(context, inner_data_type, inner_data_type)
            .map_err(Mamba2NewError::Backend)?;
        let conv_scan =
            <B::Kernels as Kernels>::Conv1dScanKernel::new(context, inner_data_type, conv_config.has_biases)
                .map_err(Mamba2NewError::Backend)?;

        let skip_connection_weight = parameter_tree
            .leaf("skip_connection_weight")?
            .validate(&[num_heads], inner_data_type)?
            .read_allocation()?;
        let ssd_update = <B::Kernels as Kernels>::SSDUpdateKernel::new(context, inner_data_type, true)
            .map_err(Mamba2NewError::Backend)?;

        let ssd_prefill = if state_dim == 64 {
            Mamba2SSDPrefillVariant::Special64(
                <B::Kernels as Kernels>::SSDPrefill64Kernel::new(context, inner_data_type)
                    .map_err(Mamba2NewError::Backend)?,
            )
        } else {
            Mamba2SSDPrefillVariant::Universal(
                <B::Kernels as Kernels>::SSDPrefillKernel::new(context, inner_data_type)
                    .map_err(Mamba2NewError::Backend)?,
            )
        };

        let out_projection = <dyn Linear<B>>::new_mixed_precision(
            inner_dim,
            [hidden_dim],
            config.has_out_biases,
            context,
            outer_data_type,
            inner_data_type,
            outer_data_type,
            &parameter_tree.subtree("out_projection")?,
        )?;

        Ok((
            Self {
                kernel_size,
                num_heads,
                num_groups,
                head_dim,
                state_dim,
                inner_dim,
                conv_dim,
                inner_data_type,
                activation_type,
                in_projection,
                gate_bias,
                split_inproj,
                conv_weight,
                conv_bias,
                conv_decode,
                conv_pack,
                conv_scan,
                skip_connection_weight,
                ssd_update,
                ssd_prefill,
                out_projection,
            },
            in_projection_input_hadamard_factors,
        ))
    }
}

impl<B: Backend> Mixer<B> for Mamba2<B> {
    fn speculation_supported(&self) -> bool {
        false
    }

    fn max_context_length(&self) -> Option<usize> {
        None
    }

    fn create_empty_state(
        &self,
        _max_context_length: Option<usize>,
        context: &B::Context,
    ) -> Result<Box<dyn MixerState<B>>, B::Error> {
        let mut conv_state = context.create_allocation(
            size_for_shape(&[self.conv_dim, self.kernel_size - 1], DataType::F32),
            AllocationType::Global,
        )?;

        let mut ssm_state = context.create_allocation(
            size_for_shape(&[self.num_heads, self.head_dim, self.state_dim], DataType::F32),
            AllocationType::Global,
        )?;

        let mut zero_encoder = Encoder::<B>::new(context)?;
        zero_encoder.encode_fill(&mut conv_state, 0);
        zero_encoder.encode_fill(&mut ssm_state, 0);
        zero_encoder.end_encoding().submit().wait_until_completed()?;

        Ok(Box::new(Mamba2State {
            conv_state,
            ssm_state,
            suffix_length: None,
        }))
    }

    fn encode(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        assert!(precalculated_rope.is_none(), "unexpected rope for mamba2 mixer");

        if !batch_dim.full_accept() {
            panic!("mamba2 doesn't support speculation");
        }

        let state = state.expect("mamba2 requires state");
        let state = state.downcast::<Mamba2State<B>>().expect("incorrect type of mamba2 state");
        let MaybeMut::Mut(state) = state else {
            panic!("mamba2 doesn't support immutable state");
        };

        assert!(state.suffix_length.is_none(), "mamba2 called with state with unaccepted tokens");

        let in_projected = self.in_projection.encode(hidden, batch_dim.size(), encoder)?;

        let mut conv_inputs =
            encoder.allocate_scratch(size_for_shape(&[batch_dim.size(), self.conv_dim], self.inner_data_type))?;
        let mut gate = encoder.allocate_scratch(size_for_shape(
            &[batch_dim.size(), self.num_heads, self.head_dim],
            self.inner_data_type,
        ))?;
        let mut time_step =
            encoder.allocate_scratch(size_for_shape(&[batch_dim.size(), self.num_heads], self.inner_data_type))?;
        self.split_inproj.encode(
            &in_projected,
            &mut conv_inputs,
            &mut gate,
            &mut time_step,
            &self.gate_bias,
            batch_dim.size() as u32,
            (self.conv_dim + self.inner_dim + self.num_heads) as u32,
            self.conv_dim as u32,
            self.inner_dim as u32,
            self.num_heads as u32,
            encoder,
        );

        let mut conv_x = encoder.allocate_scratch(size_for_shape(
            &[batch_dim.size(), self.num_heads, self.head_dim],
            self.inner_data_type,
        ))?;
        let mut state_b = encoder.allocate_scratch(size_for_shape(
            &[batch_dim.size(), self.num_groups, self.state_dim],
            self.inner_data_type,
        ))?;
        let mut state_c = encoder.allocate_scratch(size_for_shape(
            &[batch_dim.size(), self.num_groups, self.state_dim],
            self.inner_data_type,
        ))?;
        let state_stride = self.kernel_size - 1;
        if batch_dim.size() == 1 {
            self.conv_decode.encode(
                &conv_inputs,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                None::<&Allocation<B>>,
                &mut conv_x,
                &mut state_b,
                &mut state_c,
                &mut state.conv_state,
                self.kernel_size as u32,
                self.conv_dim as u32,
                state_stride as u32,
                self.conv_dim as u32,
                batch_dim.size() as u32,
                self.inner_dim as u32,
                (self.num_groups * self.state_dim) as u32,
                self.activation_type,
                encoder,
            );
        } else {
            let mut padded = encoder.allocate_scratch(size_for_shape(
                &[batch_dim.size() + state_stride, self.conv_dim],
                self.inner_data_type,
            ))?;
            self.conv_pack.encode(
                &state.conv_state,
                &conv_inputs,
                &mut padded,
                state_stride as u32,
                self.conv_dim as u32,
                batch_dim.size() as u32,
                self.conv_dim as u32,
                encoder,
            );
            self.conv_scan.encode(
                &padded,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut conv_x,
                &mut state_b,
                &mut state_c,
                &mut state.conv_state,
                batch_dim.size() as u32,
                self.kernel_size as u32,
                self.conv_dim as u32,
                state_stride as u32,
                self.conv_dim as u32,
                self.inner_dim as u32,
                (self.num_groups * self.state_dim) as u32,
                self.activation_type,
                encoder,
            );
        }

        let mut ssd_output =
            encoder.allocate_scratch(size_for_shape(&[batch_dim.size(), self.inner_dim], self.inner_data_type))?;
        let x_strides = [(self.num_heads * self.head_dim) as u32, self.head_dim as u32, 1];
        let dt_strides = [self.num_heads as u32, 1];
        let cb_strides = [(self.num_groups * self.state_dim) as u32, self.state_dim as u32, 1];
        let group_size = self.num_heads / self.num_groups;
        if batch_dim.size() == 1 {
            let state_strides = [
                (self.num_heads * self.head_dim * self.state_dim) as u32,
                (self.head_dim * self.state_dim) as u32,
                self.state_dim as u32,
                1,
            ];
            self.ssd_update.encode(
                &conv_x,
                &time_step,
                &state_b,
                &state_c,
                &self.skip_connection_weight,
                &gate,
                None::<&Allocation<B>>,
                &mut ssd_output,
                &mut state.ssm_state,
                group_size as u32,
                self.state_dim as u32,
                &x_strides,
                &dt_strides,
                &cb_strides,
                &state_strides,
                batch_dim.size() as u32,
                self.num_heads as u32,
                self.head_dim as u32,
                encoder,
            );
        } else {
            let state_strides = [(self.head_dim * self.state_dim) as u32, self.state_dim as u32, 1];
            match &self.ssd_prefill {
                Mamba2SSDPrefillVariant::Universal(ssd_prefill) => ssd_prefill.encode(
                    &conv_x,
                    &time_step,
                    &state_b,
                    &state_c,
                    &self.skip_connection_weight,
                    &gate,
                    &mut state.ssm_state,
                    &mut ssd_output,
                    batch_dim.size() as u32,
                    group_size as u32,
                    self.state_dim as u32,
                    &x_strides,
                    &dt_strides,
                    &cb_strides,
                    &state_strides,
                    self.num_heads as u32,
                    self.head_dim as u32,
                    encoder,
                ),
                Mamba2SSDPrefillVariant::Special64(ssd_prefill) => ssd_prefill.encode(
                    &conv_x,
                    &time_step,
                    &state_b,
                    &state_c,
                    &self.skip_connection_weight,
                    &gate,
                    &mut state.ssm_state,
                    &mut ssd_output,
                    batch_dim.size() as u32,
                    group_size as u32,
                    self.state_dim as u32,
                    &x_strides,
                    &dt_strides,
                    &cb_strides,
                    &state_strides,
                    self.num_heads as u32,
                    self.head_dim as u32,
                    encoder,
                ),
            }
        }

        state.suffix_length = Some(batch_dim.size());

        self.out_projection.encode(ssd_output, batch_dim.size(), encoder)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/encodable_block/mamba_mixer/ssd_prefill_test.rs"]
mod tests;
