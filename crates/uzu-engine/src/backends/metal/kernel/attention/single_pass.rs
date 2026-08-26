use std::collections::{HashMap, hash_map::Entry};

use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};

use crate::backends::{
    common::{
        Allocation, BufferArg, Encoder,
        kernel::{AttentionArguments, AttentionKernelConfig, AttentionSinglePassKernel},
    },
    metal::{Metal, context::MetalContext, error::MetalError, kernel::AttentionSinglePassMetalKernel},
};

pub struct AttentionSinglePass {
    kernels: Mutex<HashMap<bool, AttentionSinglePassMetalKernel>>,
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
    ) -> Result<MappedMutexGuard<'_, AttentionSinglePassMetalKernel>, MetalError> {
        let mut kernels = self.kernels.lock();
        if let Entry::Vacant(entry) = kernels.entry(is_trie) {
            let kernel = AttentionSinglePassMetalKernel::new(
                context,
                self.config.data_type,
                self.config.head_dim,
                self.config.has_sinks,
                self.config.is_kv_cache_ring,
                self.config.is_causal,
                is_trie,
                self.config.sliding_window_size.is_some(),
            )?;
            entry.insert(kernel);
        }
        Ok(MutexGuard::map(kernels, |kernels| kernels.get_mut(&is_trie).expect("kernel was just initialized")))
    }

    pub fn encode<'a, KT: BufferArg<'a, Metal>, VT: BufferArg<'a, Metal>>(
        &self,
        arguments: AttentionArguments<'a, Metal, KT, VT>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<Allocation<Metal>, MetalError> {
        let config = self.config;
        let mut output = encoder.allocate_constant_for_shape(
            &[arguments.suffix_length, config.num_q_heads, config.head_dim],
            config.data_type,
        )?;
        let kernel = self.get_or_create(encoder.context(), arguments.trie.is_some())?;
        kernel.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
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
            encoder,
        );
        Ok(output)
    }
}
