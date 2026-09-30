use std::collections::{HashMap, hash_map::Entry};

use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};

use crate::{
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding,
            kernel::{AttentionArguments, AttentionKernelConfig},
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            kernel::{AttentionTwoPass1MetalKernel, AttentionTwoPass2MetalKernel},
        },
    },
    data_type::DataType,
};

const PARTIAL_DATA_TYPE: DataType = DataType::F32;
const PARTIAL_BLOCKS: u32 = 32;

pub struct AttentionTwoPass {
    passes: Mutex<HashMap<bool, AttentionTwoPass1MetalKernel>>,
    second: Mutex<Option<AttentionTwoPass2MetalKernel>>,
    config: AttentionKernelConfig,
}

impl AttentionTwoPass {
    pub fn new(config: &AttentionKernelConfig) -> Self {
        Self {
            passes: Mutex::new(HashMap::new()),
            second: Mutex::new(None),
            config: *config,
        }
    }

    fn get_or_create(
        &self,
        context: &MetalContext,
        is_trie: bool,
    ) -> Result<MappedMutexGuard<'_, AttentionTwoPass1MetalKernel>, MetalError> {
        let mut passes = self.passes.lock();
        if let Entry::Vacant(entry) = passes.entry(is_trie) {
            entry.insert(AttentionTwoPass1MetalKernel::new(
                context,
                self.config.data_type,
                self.config.head_dim,
                self.config.has_sinks,
                self.config.is_kv_cache_ring,
                self.config.is_causal,
                is_trie,
                self.config.sliding_window_size.is_some(),
            )?);
        }
        Ok(MutexGuard::map(passes, |passes| passes.get_mut(&is_trie).expect("passes were just initialized")))
    }

    fn get_or_create_second(
        &self,
        context: &MetalContext,
    ) -> Result<MappedMutexGuard<'_, AttentionTwoPass2MetalKernel>, MetalError> {
        let mut second = self.second.lock();
        if second.is_none() {
            *second = Some(AttentionTwoPass2MetalKernel::new(context, self.config.data_type, self.config.head_dim)?);
        }
        Ok(MutexGuard::map(second, |second| second.as_mut().expect("pass was just initialized")))
    }

    pub fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        let config = self.config;
        let mut partials = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, PARTIAL_BLOCKS, config.head_dim],
            PARTIAL_DATA_TYPE,
        )?;
        let mut sums = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, PARTIAL_BLOCKS],
            PARTIAL_DATA_TYPE,
        )?;
        let mut maxs = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, PARTIAL_BLOCKS],
            PARTIAL_DATA_TYPE,
        )?;
        let first = self.get_or_create(command_buffer.context(), arguments.trie.is_some())?;
        first.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut partials,
            &mut sums,
            &mut maxs,
            config.num_q_heads / config.num_groups,
            arguments.cache.prefix_len() + arguments.suffix_length,
            config.head_dim,
            config.num_groups * config.head_dim,
            config.head_dim,
            config.num_groups * config.head_dim,
            arguments.cache.ring_params(),
            config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()),
            config.num_q_heads,
            arguments.suffix_length,
            arguments.trie,
            config.sliding_window_size,
            arguments.sinks,
            command_buffer,
        );
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, config.head_dim],
            config.data_type,
        )?;
        let second = self.get_or_create_second(command_buffer.context())?;
        second.encode(
            &partials,
            &sums,
            &maxs,
            &mut output,
            config.num_q_heads,
            arguments.suffix_length,
            command_buffer,
        );
        Ok(output)
    }
}
