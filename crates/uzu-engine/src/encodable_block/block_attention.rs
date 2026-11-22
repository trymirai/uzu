//! Attention residuals over blocks of layers; see `BlockAttentionResidualConfig`.
//!
//! The stream is slot-major, `[slot][token][channel]` in the model dtype: one slot per completed block, then the
//! partial sum of the current block, then the message bank of a repeated loop pass.

use std::range::Range;

use bytemuck::{AnyBitPattern, NoUninit};
use half::bf16;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu,
        Kernels,
        kernel::{DepthAttentionKernel, MessageGateApplyKernel, MessageGateDownKernel, TensorAddScaleKernel},
    },
    config::{residual::block_attention::BlockAttentionResidualConfig, transformer::LoopConfig},
    data_type::{DataType, DataTypeRepr},
    encodable_block::residual::{ResidualEncodeError, ResidualTarget, SublayerUpdate},
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::{maybe_owned::MaybeOwned, size_for_shape::size_for_shape},
};

#[derive(Debug, Error)]
pub enum BlockAttentionNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("Looped layers need a message gate config")]
    MissingMessageGateConfig,
    #[error("The first mixer reads the embeddings directly, so its inputs share one read")]
    SeparateReadsOfEmbeddings,
}

#[derive(Debug, Error)]
pub enum BlockAttentionEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
}

/// Which sublayer a residual connection feeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SublayerRole {
    Mixer,
    Mlp,
}

/// Where a residual connection sits. `num_reads` is 1 when all `num_inputs` share one read of the stream.
#[derive(Debug, Clone)]
pub struct ConnectionPlacement {
    pub layer_index: u32,
    pub num_layers: u32,
    pub role: SublayerRole,
    pub num_inputs: u32,
    pub num_reads: u32,
    pub loop_config: Option<LoopConfig>,
}

impl ConnectionPlacement {
    pub fn is_looped(&self) -> bool {
        self.loop_config.as_ref().is_some_and(|loop_config| loop_config.is_looped(self.layer_index))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BlockAttentionLayout {
    pub model_dim: u32,
    pub block_size: u32,
    pub max_blocks: u32,
    pub bank_size: u32,
}

impl BlockAttentionLayout {
    pub fn new(
        config: &BlockAttentionResidualConfig,
        model_dim: u32,
        num_layers: u32,
        loop_config: Option<&LoopConfig>,
    ) -> Self {
        let block_size = config.block_size;
        Self {
            model_dim,
            block_size,
            max_blocks: num_layers.div_ceil(block_size),
            // The blocks a pass stores after its first one, and its output.
            bank_size: loop_config
                .map(|loop_config| (loop_config.end_layer - loop_config.start_layer) / block_size)
                .unwrap_or(0),
        }
    }

    pub fn partial_slot(&self) -> u32 {
        self.max_blocks
    }

    pub fn bank_slot(
        &self,
        index: u32,
    ) -> u32 {
        self.max_blocks + 1 + index
    }

    pub fn slot_count(&self) -> u32 {
        self.max_blocks + 1 + self.bank_size
    }

    /// The bytes of `slot` in a stream of `token_count` tokens.
    pub fn slot_range(
        &self,
        slot: u32,
        token_count: u32,
        data_type: DataType,
    ) -> Range<usize> {
        let slot_size = size_for_shape(&[token_count, self.model_dim], data_type);
        Range::from(slot as usize * slot_size..(slot as usize + 1) * slot_size)
    }

    /// The completed blocks a sublayer sees: those stored by the mixers of earlier block-starting layers, and by
    /// its own layer's mixer for the MLP.
    fn blocks_before(
        &self,
        layer_index: u32,
        role: SublayerRole,
    ) -> u32 {
        match role {
            SublayerRole::Mixer => layer_index.div_ceil(self.block_size),
            SublayerRole::Mlp => layer_index / self.block_size + 1,
        }
    }

    fn read_slots(
        &self,
        layer_index: u32,
        role: SublayerRole,
    ) -> Box<[u32]> {
        (0..self.blocks_before(layer_index, role)).chain([self.partial_slot()]).collect()
    }

    fn bank_slots(&self) -> Box<[u32]> {
        (0..self.bank_size).map(|index| self.bank_slot(index)).collect()
    }
}

/// Softmax mixes of normalized stream slots with learned queries.
struct DepthRead<B: Backend> {
    queries: B::GlobalGpuBuffer,
    query_count: u32,
    slots: Box<[u32]>,
    kernel: <B::Kernels as Kernels>::DepthAttentionKernel,
}

impl<B: Backend> DepthRead<B> {
    async fn new(
        layout: &BlockAttentionLayout,
        data_type: DataType,
        query_count: u32,
        slots: Box<[u32]>,
        queries_name: &str,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, BlockAttentionNewError<B>> {
        // The kernel keeps 8 channels of every source per thread, in one threadgroup of 512 threads.
        assert!(
            layout.model_dim.is_multiple_of(8) && layout.model_dim <= 4096,
            "depth attention needs a model dimension that is a multiple of 8 and at most 4096, got {}",
            layout.model_dim
        );
        assert!(slots.len() <= 8 && query_count <= 4, "depth attention mixes at most 8 sources with at most 4 queries");
        let queries = parameter_tree.read_buffer(queries_name, &[query_count, layout.model_dim], DataType::F32).await?;
        let kernel = <B::Kernels as Kernels>::DepthAttentionKernel::new(context, data_type, layout.model_dim, query_count)
            .map_err(BlockAttentionNewError::Backend)?;
        Ok(Self {
            queries,
            query_count,
            slots,
            kernel,
        })
    }

    /// Mixes rows `token_offset..token_offset + token_count` of a stream of `stream_token_count` tokens into
    /// `[query][token][channel]`.
    fn encode(
        &self,
        layout: &BlockAttentionLayout,
        data_type: DataType,
        stream: impl BufferRef<Backend = B>,
        stream_token_count: u32,
        token_offset: u32,
        token_count: u32,
        epsilon: f32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let slots = command_buffer.allocate_constant_from_slice(&self.slots)?;
        let mut output =
            command_buffer.allocate_scratch_for_shape(&[self.query_count, token_count, layout.model_dim], data_type)?;
        self.kernel.encode(
            stream,
            &slots,
            &self.queries,
            &mut output,
            token_count,
            stream_token_count,
            token_offset,
            self.slots.len() as u32,
            epsilon,
            command_buffer,
        );
        Ok(output)
    }
}

const MAX_MESSAGE_GATE_RANK: u32 = 128;

/// Copies host values into a new GPU buffer.
pub(crate) fn upload<B: Backend, T: NoUninit + AnyBitPattern>(
    values: &[T],
    context: &B::Context,
) -> Result<B::GlobalGpuBuffer, B::Error> {
    let mut buffer = context.create_buffer_cpu_gpu(size_of_val(values))?;
    buffer.as_slice_mut::<T>().copy_from_slice(values);
    buffer.handoff()
}

/// The message gates of every input slot of a connection, see `MessageGateConfig`: per slot, z = down · [h / rms(h);
/// m / rms(m)] and h' = (1 - tanh(F z)) h + tanh(G z) m.
struct MessageGates<B: Backend> {
    slot_count: u32,
    rank: u32,
    /// `[slot][rank][2 * model_dim]` bfloat16.
    down_weights: B::GlobalGpuBuffer,
    /// `[slot][model_dim][rank]` float32.
    forget_weights: B::GlobalGpuBuffer,
    gain_weights: B::GlobalGpuBuffer,
    down_kernel: <B::Kernels as Kernels>::MessageGateDownKernel,
    apply_kernel: <B::Kernels as Kernels>::MessageGateApplyKernel,
}

impl<B: Backend> MessageGates<B> {
    async fn new(
        model_dim: u32,
        rank: u32,
        slot_count: u32,
        data_type: DataType,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, BlockAttentionNewError<B>> {
        // The gate kernel keeps each lane's share of the low-rank projection in registers.
        assert!(rank <= MAX_MESSAGE_GATE_RANK, "message gates of rank {rank} exceed {MAX_MESSAGE_GATE_RANK}");
        assert!(
            model_dim.is_multiple_of(8) && model_dim <= 4096,
            "message gates read 8 channels at a time, up to a model dimension of 4096, got {model_dim}"
        );
        let mut down = Vec::new();
        let mut forget = Vec::new();
        let mut gain = Vec::new();
        for slot in 0..slot_count {
            let tree = parameter_tree.subtree(&slot.to_string());
            let down_tree = tree.subtree("down_projection").subtree("weights");
            down.extend_from_slice(&down_tree.read_slice::<bf16>("weights", &[rank, 2 * model_dim]).await?);
            let forget_tree = tree.subtree("forget_projection").subtree("weights");
            forget.extend_from_slice(&forget_tree.read_slice::<f32>("weights", &[model_dim, rank]).await?);
            let gain_tree = tree.subtree("gain_projection").subtree("weights");
            gain.extend_from_slice(&gain_tree.read_slice::<f32>("weights", &[model_dim, rank]).await?);
        }
        let backend = BlockAttentionNewError::Backend;
        Ok(Self {
            slot_count,
            rank,
            down_weights: upload::<B, _>(&down, context).map_err(backend)?,
            forget_weights: upload::<B, _>(&forget, context).map_err(backend)?,
            gain_weights: upload::<B, _>(&gain, context).map_err(backend)?,
            down_kernel: <B::Kernels as Kernels>::MessageGateDownKernel::new(
                context, data_type, model_dim, rank, slot_count,
            )
            .map_err(backend)?,
            apply_kernel: <B::Kernels as Kernels>::MessageGateApplyKernel::new(
                context, data_type, model_dim, rank, slot_count,
            )
            .map_err(backend)?,
        })
    }

    /// Gates the messages of every slot into its input. `inputs` are `input_slot_stride` elements apart, zero when the
    /// slots share one input; `messages` and the output are slot-major.
    fn encode(
        &self,
        model_dim: u32,
        data_type: DataType,
        inputs: impl BufferRef<Backend = B>,
        input_slot_stride: u32,
        messages: impl BufferRef<Backend = B>,
        token_count: u32,
        epsilon: f32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut low_rank =
            command_buffer.allocate_scratch_for_shape(&[self.slot_count, token_count, self.rank], f32::data_type())?;
        let mut output = command_buffer.allocate_scratch_for_shape(&[self.slot_count, token_count, model_dim], data_type)?;
        self.down_kernel.encode(
            inputs,
            messages,
            &self.down_weights,
            &mut low_rank,
            token_count,
            input_slot_stride,
            epsilon,
            command_buffer,
        );
        self.apply_kernel.encode(
            inputs,
            messages,
            &low_rank,
            &self.forget_weights,
            &self.gain_weights,
            &mut output,
            token_count,
            input_slot_stride,
            command_buffer,
        );
        Ok(output)
    }
}

struct LoopMessages<B: Backend> {
    read: DepthRead<B>,
    gates: MessageGates<B>,
}

pub struct BlockAttentionConnection<B: Backend> {
    layout: BlockAttentionLayout,
    epsilon: f32,
    data_type: DataType,
    num_inputs: u32,
    /// None for the first mixer, which reads the embeddings directly.
    read: Option<DepthRead<B>>,
    messages: Option<LoopMessages<B>>,
    /// The block slot that a block-starting mixer stores the partial sum in.
    stored_block_slot: Option<u32>,
    residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel,
}

impl<B: Backend> BlockAttentionConnection<B> {
    pub async fn new(
        config: &BlockAttentionResidualConfig,
        placement: &ConnectionPlacement,
        model_dim: u32,
        data_type: DataType,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, BlockAttentionNewError<B>> {
        let layout = BlockAttentionLayout::new(config, model_dim, placement.num_layers, placement.loop_config.as_ref());
        let reads_stream_directly = placement.role == SublayerRole::Mixer && placement.layer_index == 0;
        let read = if reads_stream_directly {
            if placement.num_reads != 1 {
                return Err(BlockAttentionNewError::SeparateReadsOfEmbeddings);
            }
            None
        } else {
            Some(
                DepthRead::new(
                    &layout,
                    data_type,
                    placement.num_reads,
                    layout.read_slots(placement.layer_index, placement.role),
                    "queries",
                    parameter_tree,
                    context,
                )
                .await?,
            )
        };
        let messages = if placement.is_looped() {
            let gate_config =
                config.message_gate_config.as_ref().ok_or(BlockAttentionNewError::MissingMessageGateConfig)?;
            let read = DepthRead::new(
                &layout,
                data_type,
                placement.num_inputs,
                layout.bank_slots(),
                "message_queries",
                parameter_tree,
                context,
            )
            .await?;
            let gates = MessageGates::new(
                model_dim,
                gate_config.rank,
                placement.num_inputs,
                data_type,
                &parameter_tree.subtree("message_gates"),
                context,
            )
            .await?;
            Some(LoopMessages {
                read,
                gates,
            })
        } else {
            None
        };
        let starts_block =
            placement.role == SublayerRole::Mixer && placement.layer_index.is_multiple_of(config.block_size);
        Ok(Self {
            layout,
            epsilon: config.epsilon,
            data_type,
            num_inputs: placement.num_inputs,
            read,
            messages,
            stored_block_slot: starts_block.then_some(placement.layer_index / config.block_size),
            // In place on the partial sum, with the update as the added operand.
            residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, true, true)
                .map_err(BlockAttentionNewError::Backend)?,
        })
    }

    fn encode_inputs(
        &self,
        stream: &B::ScratchBuffer,
        token_count: u32,
        has_messages: bool,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, BlockAttentionEncodeError<B>> {
        let model_dim = self.layout.model_dim;
        let input_size = size_for_shape(&[token_count, model_dim], self.data_type);
        let mixes = match &self.read {
            Some(read) => read
                .encode(&self.layout, self.data_type, stream, token_count, 0, token_count, self.epsilon, command_buffer)
                .map_err(BlockAttentionEncodeError::Backend)?,
            None => {
                let mut mixes =
                    command_buffer.allocate_scratch(input_size).map_err(BlockAttentionEncodeError::Backend)?;
                let partial = self.layout.slot_range(self.layout.partial_slot(), token_count, self.data_type);
                command_buffer.encode_copy(stream.subrange(partial), &mut mixes);
                mixes
            },
        };
        // A single read serves every input: the body gets it once, unless messages make the inputs differ.
        let is_shared = self.read.as_ref().map_or(1, |read| read.query_count) != self.num_inputs;
        if !has_messages {
            return Ok(mixes);
        }
        {
            let command_buffer = &mut *command_buffer.debug_group("message gates");
            let messages = self.messages.as_ref().expect("this connection received loop messages but has no gates");
            let message_mixes = messages
                .read
                .encode(&self.layout, self.data_type, stream, token_count, 0, token_count, self.epsilon, command_buffer)
                .map_err(BlockAttentionEncodeError::Backend)?;
            let input_slot_stride = if is_shared {
                0
            } else {
                token_count * model_dim
            };
            messages
                .gates
                .encode(
                    model_dim,
                    self.data_type,
                    &mixes,
                    input_slot_stride,
                    &message_mixes,
                    token_count,
                    self.epsilon,
                    command_buffer,
                )
                .map_err(BlockAttentionEncodeError::Backend)
        }
    }

    /// Runs `body` on the inputs this connection reads and adds its output to the stream, in place.
    pub fn encode<E: From<ResidualEncodeError<B>>>(
        &self,
        mut stream: B::ScratchBuffer,
        token_count: u32,
        has_messages: bool,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        body: impl FnOnce(
            MaybeOwned<'_, B::ScratchBuffer>,
            Option<ResidualTarget<'_, B>>,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<SublayerUpdate<B>, E>,
    ) -> Result<B::ScratchBuffer, E> {
        let error = |error: BlockAttentionEncodeError<B>| E::from(ResidualEncodeError::BlockAttention(error));
        let inputs = {
            let command_buffer = &mut *command_buffer.debug_group("block read");
            self.encode_inputs(&stream, token_count, has_messages, command_buffer).map_err(error)?
        };
        let partial = self.layout.slot_range(self.layout.partial_slot(), token_count, self.data_type);
        if let Some(block_slot) = self.stored_block_slot {
            // The reads are done, so the partial sum moves into its block before the body replaces it.
            let block = self.layout.slot_range(block_slot, token_count, self.data_type);
            assert!(block.end <= partial.start, "a block slot follows the partial sum");
            let (blocks, rest) = (&mut stream).split_at(partial.start);
            command_buffer.encode_copy(
                rest.as_ref().subrange(0..partial.iter().len()),
                blocks.subrange_mut(block.start..block.end),
            );
        }
        let target = ResidualTarget {
            stream: &mut stream,
            range: partial.start..partial.end,
            accumulate: self.stored_block_slot.is_none(),
        };
        let update = match body(MaybeOwned::Owned(inputs), Some(target), command_buffer)? {
            SublayerUpdate::Returned(update) => update,
            SublayerUpdate::Written => return Ok(stream),
        };
        let command_buffer = &mut *command_buffer.debug_group("block write");
        match self.stored_block_slot {
            Some(_) => {
                command_buffer.encode_copy(&update, (&mut stream).subrange_mut(partial));
            },
            None => {
                let elements = token_count * self.layout.model_dim;
                self.residual_add.encode(
                    None::<&B::ScratchBuffer>,
                    Some(&update),
                    (&mut stream).subrange_mut(partial),
                    elements,
                    elements,
                    1.0,
                    command_buffer,
                );
            },
        }
        Ok(stream)
    }
}

pub struct BlockAttentionHead<B: Backend> {
    layout: BlockAttentionLayout,
    epsilon: f32,
    data_type: DataType,
    read: DepthRead<B>,
}

impl<B: Backend> BlockAttentionHead<B> {
    pub async fn new(
        config: &BlockAttentionResidualConfig,
        model_dim: u32,
        num_layers: u32,
        loop_config: Option<&LoopConfig>,
        data_type: DataType,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, BlockAttentionNewError<B>> {
        let layout = BlockAttentionLayout::new(config, model_dim, num_layers, loop_config);
        let slots = (0..layout.max_blocks).chain([layout.partial_slot()]).collect();
        let read = DepthRead::new(&layout, data_type, 1, slots, "query", parameter_tree, context).await?;
        Ok(Self {
            layout,
            epsilon: config.epsilon,
            data_type,
            read,
        })
    }

    pub fn layout(&self) -> &BlockAttentionLayout {
        &self.layout
    }

    /// The bytes of the partial sum in a stream of `token_count` tokens.
    pub fn partial_rows(
        &self,
        token_count: u32,
    ) -> std::ops::Range<usize> {
        let partial = self.layout.slot_range(self.layout.partial_slot(), token_count, self.data_type);
        partial.start..partial.end
    }

    pub fn expand(
        &self,
        embeddings: B::ScratchBuffer,
        token_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut stream = command_buffer.allocate_scratch(
            size_for_shape(&[self.layout.slot_count(), token_count, self.layout.model_dim], self.data_type),
        )?;
        let partial = self.layout.slot_range(self.layout.partial_slot(), token_count, self.data_type);
        command_buffer.encode_copy(&embeddings, (&mut stream).subrange_mut(partial));
        Ok(stream)
    }

    pub fn readout(
        &self,
        stream: impl BufferRef<Backend = B>,
        token_count: u32,
        row_offset: u32,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        self.read.encode(
            &self.layout,
            self.data_type,
            stream,
            token_count,
            row_offset,
            row_count,
            self.epsilon,
            command_buffer,
        )
    }

    /// The stream entering a repeated loop pass: the loop's input, with the previous pass's blocks after its first
    /// one and its output as the message bank.
    pub fn begin_loop_pass(
        &self,
        loop_input: &B::ScratchBuffer,
        previous_pass_output: &B::ScratchBuffer,
        num_loop_input_blocks: u32,
        token_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let layout = &self.layout;
        let mut stream = command_buffer.allocate_scratch(loop_input.size())?;
        command_buffer.encode_copy(loop_input, &mut stream);
        let bank_sources = (num_loop_input_blocks + 1..num_loop_input_blocks + layout.bank_size)
            .chain([layout.partial_slot()]);
        for (bank_index, source_slot) in bank_sources.enumerate() {
            command_buffer.encode_copy(
                previous_pass_output.subrange(layout.slot_range(source_slot, token_count, self.data_type)),
                (&mut stream).subrange_mut(layout.slot_range(
                    layout.bank_slot(bank_index as u32),
                    token_count,
                    self.data_type,
                )),
            );
        }
        Ok(stream)
    }
}
