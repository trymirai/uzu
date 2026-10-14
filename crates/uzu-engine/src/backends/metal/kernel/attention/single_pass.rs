use std::collections::{HashMap, hash_map::Entry};

use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};

use crate::backends::{
    common::{
        Backend, BufferRef, CommandBufferEncoding,
        kernel::{AttentionArguments, AttentionKernelConfig, AttentionSinglePassKernel},
    },
    metal::{
        Metal, command_buffer::MetalCommandBufferEncoding, context::MetalContext, error::MetalError,
        kernel::AttentionSinglePassMetalKernel,
    },
};

pub struct AttentionSinglePass {
    kernels: Mutex<HashMap<(bool, bool), AttentionSinglePassMetalKernel>>,
    config: AttentionKernelConfig,
}

impl AttentionSinglePass {
    pub fn new(config: &AttentionKernelConfig) -> Self {
        Self {
            kernels: Mutex::new(HashMap::new()),
            config: *config,
        }
    }

    fn get_or_create(
        &self,
        context: &MetalContext,
        is_trie: bool,
        has_rope: bool,
    ) -> Result<MappedMutexGuard<'_, AttentionSinglePassMetalKernel>, MetalError> {
        let mut kernels = self.kernels.lock();
        let key = (is_trie, has_rope);
        if let Entry::Vacant(entry) = kernels.entry(key) {
            let kernel = AttentionSinglePassMetalKernel::new(
                context,
                self.config.data_type,
                self.config.head_dim,
                self.config.has_sinks,
                self.config.is_kv_cache_ring,
                self.config.is_causal,
                is_trie,
                self.config.sliding_window_size.is_some(),
                has_rope,
            )?;
            entry.insert(kernel);
        }
        Ok(MutexGuard::map(kernels, |kernels| kernels.get_mut(&key).expect("kernel was just initialized")))
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
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        let config = self.config;
        if let Some(rope) = arguments.rope {
            assert!(
                rope.dim > 0 && rope.dim.is_multiple_of(2) && rope.dim <= config.head_dim,
                "invalid RoPE dimension"
            );
        }
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, config.head_dim],
            config.data_type,
        )?;
        let kernel =
            self.get_or_create(command_buffer.context(), arguments.trie.is_some(), arguments.rope.is_some())?;
        kernel.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            arguments.rope.map(|rope| &rope.cosines),
            arguments.rope.map(|rope| &rope.sines),
            &mut output,
            config.num_q_heads / config.num_groups,
            arguments.cache.prefix_len() + arguments.suffix_length,
            config.head_dim,
            config.num_groups * config.head_dim,
            config.head_dim,
            config.num_groups * config.head_dim,
            arguments.cache.ring_params(),
            config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()),
            arguments.trie,
            config.sliding_window_size,
            arguments.sinks,
            config.num_q_heads,
            arguments.suffix_length,
            arguments.rope.map(|rope| rope.dim),
            arguments.rope.map(|_| arguments.cache.context_length()),
            command_buffer,
        );
        Ok(output)
    }
}
