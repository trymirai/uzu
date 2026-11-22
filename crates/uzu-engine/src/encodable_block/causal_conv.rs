use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context,
        kernel::{CausalConvKernel, CausalConvShiftKernel, CausalConvTokensKernel, Kernels},
    },
    config::token_mixer::convolutions::CausalConvConfig,
    data_type::DataType,
    encodable_block::batch_topology::BatchTopology,
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::size_for_shape::size_for_shape,
};

/// The kernel keeps the window of each channel in registers.
const MAX_KERNEL_SIZE: u32 = 8;

/// Flat batches up to this many tokens convolve them one after another in each channel's thread; longer ones get a
/// thread per token and channel.
const SEQUENTIAL_TOKENS: u32 = 8;

#[derive(Debug, Error)]
pub enum CausalConvNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("Causal convolutions support kernel sizes from 2 to {MAX_KERNEL_SIZE}, got {0}")]
    UnsupportedKernelSize(u32),
    #[error("{groups} dynamic groups do not divide {channels} channels")]
    InvalidDynamicGroups {
        groups: u32,
        channels: u32,
    },
}

/// The inputs a trie batch left at each of its nodes, waiting for one of them to be accepted.
enum PendingInputs<B: Backend> {
    Flat,
    Trie(B::ScratchBuffer),
}

/// The last `kernel_size - 1` inputs of each channel, in float32.
pub struct CausalConvState<B: Backend> {
    inputs: B::GlobalGpuBuffer,
    pending: Option<PendingInputs<B>>,
    is_zeroed: bool,
}

impl<B: Backend> CausalConvState<B> {
    /// The inputs for a kernel that convolves with this state itself, zeroed on first use; `finish_external` records
    /// what the batch leaves to accept.
    pub fn begin_external(
        &mut self,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> &mut B::GlobalGpuBuffer {
        assert!(self.pending.is_none(), "a causal convolution state was used before accepting its tokens");
        if !self.is_zeroed {
            command_buffer.encode_fill(&mut self.inputs, 0);
            self.is_zeroed = true;
        }
        &mut self.inputs
    }

    /// Records that a flat batch updated the state, or the inputs a trie batch left at each node.
    pub fn finish_external(
        &mut self,
        node_inputs: Option<B::ScratchBuffer>,
    ) {
        self.pending = Some(match node_inputs {
            Some(node_inputs) => PendingInputs::Trie(node_inputs),
            None => PendingInputs::Flat,
        });
    }

    pub fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let accepted_index = *accepted_indices.last().expect("a causal convolution accepted no tokens");
        match self.pending.take().expect("a causal convolution state has nothing to accept") {
            PendingInputs::Flat => {},
            PendingInputs::Trie(node_inputs) => {
                let node_size = self.inputs.size();
                let start = accepted_index as usize * node_size;
                command_buffer.encode_copy(node_inputs.subrange(start..start + node_size), &mut self.inputs);
            },
        }
    }
}

/// A stateful depthwise causal convolution; see `CausalConvConfig`.
pub struct CausalConv<B: Backend> {
    channels: u32,
    kernel_size: u32,
    group_size: u32,
    has_identity: bool,
    has_taps: bool,
    output_scale: f32,
    data_type: DataType,
    weights: B::GlobalGpuBuffer,
    flat_kernel: <B::Kernels as Kernels>::CausalConvKernel,
    trie_kernel: <B::Kernels as Kernels>::CausalConvKernel,
    tokens_kernel: <B::Kernels as Kernels>::CausalConvTokensKernel,
    shift_kernel: <B::Kernels as Kernels>::CausalConvShiftKernel,
}

impl<B: Backend> CausalConv<B> {
    pub async fn new(
        channels: u32,
        data_type: DataType,
        config: &CausalConvConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, CausalConvNewError<B>> {
        let kernel_size = config.kernel_size;
        if !(2..=MAX_KERNEL_SIZE).contains(&kernel_size) {
            return Err(CausalConvNewError::UnsupportedKernelSize(kernel_size));
        }
        let group_size = match config.num_dynamic_groups {
            Some(groups) if groups == 0 || channels % groups != 0 => {
                return Err(CausalConvNewError::InvalidDynamicGroups {
                    groups,
                    channels,
                });
            },
            Some(groups) => channels / groups,
            None => channels,
        };
        let has_taps = config.num_dynamic_groups.is_some();
        let weights = parameter_tree.read_buffer("weights", &[channels, kernel_size], DataType::F32).await?;
        let new_kernel = |is_trie: bool| {
            <B::Kernels as Kernels>::CausalConvKernel::new(
                context,
                data_type,
                channels,
                kernel_size,
                group_size,
                config.has_identity,
                has_taps,
                is_trie,
            )
            .map_err(CausalConvNewError::Backend)
        };
        Ok(Self {
            channels,
            kernel_size,
            group_size,
            has_identity: config.has_identity,
            has_taps,
            output_scale: config.output_scale.unwrap_or(1.0),
            data_type,
            weights,
            flat_kernel: new_kernel(false)?,
            trie_kernel: new_kernel(true)?,
            tokens_kernel: <B::Kernels as Kernels>::CausalConvTokensKernel::new(
                context,
                data_type,
                channels,
                kernel_size,
                group_size,
                config.has_identity,
                has_taps,
            )
            .map_err(CausalConvNewError::Backend)?,
            shift_kernel: <B::Kernels as Kernels>::CausalConvShiftKernel::new(context, data_type, channels, kernel_size)
                .map_err(CausalConvNewError::Backend)?,
        })
    }

    pub fn kernel_size(&self) -> u32 {
        self.kernel_size
    }

    pub fn group_size(&self) -> u32 {
        self.group_size
    }

    pub fn has_identity(&self) -> bool {
        self.has_identity
    }

    pub fn has_taps(&self) -> bool {
        self.has_taps
    }

    pub fn output_scale(&self) -> f32 {
        self.output_scale
    }

    /// `[channel][kernel_size]` float32 weights; the last tap weights the current token.
    pub fn weights(&self) -> &B::GlobalGpuBuffer {
        &self.weights
    }

    /// Moves the last inputs of a flat batch, whose rows are `input_stride` elements apart, into `state`.
    pub fn encode_shift(
        &self,
        input: impl BufferRef<Backend = B>,
        input_stride: u32,
        token_count: u32,
        state: &mut B::GlobalGpuBuffer,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        self.shift_kernel.encode(input, state, token_count, input_stride, command_buffer);
    }

    pub fn create_state(
        &self,
        context: &B::Context,
    ) -> Result<CausalConvState<B>, B::Error> {
        Ok(CausalConvState {
            inputs: context.create_buffer_gpu(size_for_shape(&[self.channels, self.kernel_size - 1], DataType::F32))?,
            pending: None,
            is_zeroed: false,
        })
    }

    /// Convolves the batch's rows of `input`, which are `input_stride` elements apart. `taps` holds each token's
    /// dynamic taps, `taps_stride` elements apart. Without a state, the convolution starts from zeros.
    pub fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        input_stride: u32,
        taps: Option<(impl BufferRef<Backend = B>, u32)>,
        batch_dim: &BatchTopology<B>,
        state: Option<&mut CausalConvState<B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output = command_buffer.allocate_scratch_for_shape(&[batch_dim.size(), self.channels], self.data_type)?;
        self.encode_into(input, input_stride, taps, batch_dim, state, &mut output, false, command_buffer)?;
        Ok(output)
    }

    /// Like `encode`, but writes the `[token][channel]` outputs into `output`, or adds them to it with `accumulate`.
    pub fn encode_into(
        &self,
        input: impl BufferRef<Backend = B>,
        input_stride: u32,
        taps: Option<(impl BufferRef<Backend = B>, u32)>,
        batch_dim: &BatchTopology<B>,
        state: Option<&mut CausalConvState<B>>,
        mut output: impl BufferMut<Backend = B>,
        accumulate: bool,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let token_count = batch_dim.size();
        let accumulate = u32::from(accumulate);
        let (taps, taps_stride) = match taps {
            Some((buffer, stride)) => (Some(buffer), stride),
            None => (None, 0),
        };

        let Some(state) = state else {
            assert!(batch_dim.full_accept(), "a causal convolution needs a state to run on a trie");
            let mut zeros =
                command_buffer.allocate_scratch(size_for_shape(&[self.channels, self.kernel_size - 1], DataType::F32))?;
            command_buffer.encode_fill(&mut zeros, 0);
            self.tokens_kernel.encode(
                input,
                taps,
                &self.weights,
                &zeros,
                output.reborrow(),
                token_count,
                input_stride,
                taps_stride,
                self.output_scale,
                accumulate,
                command_buffer,
            );
            return Ok(());
        };

        assert!(state.pending.is_none(), "a causal convolution state was used before accepting its tokens");
        if !state.is_zeroed {
            command_buffer.encode_fill(&mut state.inputs, 0);
            state.is_zeroed = true;
        }
        match batch_dim.nodes() {
            None if token_count > SEQUENTIAL_TOKENS => {
                self.tokens_kernel.encode(
                    input,
                    taps,
                    &self.weights,
                    &state.inputs,
                    output.reborrow(),
                    token_count,
                    input_stride,
                    taps_stride,
                    self.output_scale,
                    accumulate,
                    command_buffer,
                );
                self.shift_kernel.encode(input, &mut state.inputs, token_count, input_stride, command_buffer);
                state.pending = Some(PendingInputs::Flat);
            },
            None => {
                self.flat_kernel.encode(
                    input,
                    taps,
                    &self.weights,
                    &mut state.inputs,
                    None::<&B::GlobalGpuBuffer>,
                    None::<&mut B::ScratchBuffer>,
                    output.reborrow(),
                    token_count,
                    input_stride,
                    taps_stride,
                    self.output_scale,
                    accumulate,
                    command_buffer,
                );
                state.pending = Some(PendingInputs::Flat);
            },
            Some(trie) => {
                let mut node_inputs = command_buffer.allocate_scratch_for_shape(
                    &[token_count, self.channels, self.kernel_size - 1],
                    DataType::F32,
                )?;
                self.trie_kernel.encode(
                    input,
                    taps,
                    &self.weights,
                    &mut state.inputs,
                    Some(trie),
                    Some(&mut node_inputs),
                    output.reborrow(),
                    token_count,
                    input_stride,
                    taps_stride,
                    self.output_scale,
                    accumulate,
                    command_buffer,
                );
                state.pending = Some(PendingInputs::Trie(node_inputs));
            },
        }
        Ok(())
    }
}
