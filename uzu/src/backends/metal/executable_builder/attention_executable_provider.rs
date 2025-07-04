use std::{collections::HashMap, rc::Rc};

use mpsgraph::Executable;
use objc2::rc::Retained;

use crate::{
    backends::metal::{
        MTLContext, compilation_parameters::CompilationConfig,
        forward_pass::transformer_layer::attention_executable,
    },
    config::DecoderConfig,
};

pub struct AttentionExecutableProviderConfig {
    pub prefix_length_step_by_suffix_length: HashMap<usize, usize>,
}

impl AttentionExecutableProviderConfig {
    pub fn new(
        prefix_length_step_by_suffix_length: HashMap<usize, usize>
    ) -> Self {
        Self {
            prefix_length_step_by_suffix_length,
        }
    }
}

pub struct AttentionExecutableProvider {
    mtl_context: Rc<MTLContext>,
    decoder_config: Rc<DecoderConfig>,
    compilation_config: Rc<CompilationConfig>,
    config: AttentionExecutableProviderConfig,
    pub executables: HashMap<usize, HashMap<usize, Retained<Executable>>>,
}

impl AttentionExecutableProvider {
    pub fn new(
        mtl_context: Rc<MTLContext>,
        decoder_config: Rc<DecoderConfig>,
        compilation_config: Rc<CompilationConfig>,
        config: AttentionExecutableProviderConfig,
    ) -> Self {
        let max_prefix_length: usize;
        if let Some(sliding_window_sizes) = &decoder_config.sliding_window_sizes
        {
            max_prefix_length = sliding_window_sizes
                .iter()
                .map(|size| size.unwrap_or(0))
                .max()
                .unwrap_or(0);
        } else {
            max_prefix_length = decoder_config.context_length;
        }

        let mut instance = Self {
            mtl_context: mtl_context,
            decoder_config,
            compilation_config,
            config,
            executables: HashMap::new(),
        };
        instance.prefetch(max_prefix_length);
        instance
    }

    fn prefetch(
        &mut self,
        max_prefix_length: usize,
    ) {
        let prefix_length_step_by_suffix_length =
            self.config.prefix_length_step_by_suffix_length.clone();
        for (suffix_length, prefix_length_step) in
            prefix_length_step_by_suffix_length
        {
            let _ = self.executable(suffix_length, 0);
            if prefix_length_step == 0 {
                continue;
            }

            for prefix_length in (prefix_length_step..=max_prefix_length)
                .step_by(prefix_length_step)
            {
                let _ = self.executable(suffix_length, prefix_length);
            }
        }
    }

    pub fn executable(
        &mut self,
        suffix_length: usize,
        prefix_length: usize,
    ) -> &Retained<Executable> {
        if !self.executables.contains_key(&suffix_length) {
            self.executables.insert(suffix_length, HashMap::new());
        }

        let need_to_compile = {
            let executables_for_suffix =
                self.executables.get(&suffix_length).unwrap();
            !executables_for_suffix.contains_key(&prefix_length)
        };

        if need_to_compile {
            let executable = self.compile(suffix_length, prefix_length);
            let executables_for_suffix =
                self.executables.get_mut(&suffix_length).unwrap();
            executables_for_suffix.insert(prefix_length, executable);
        }

        &self
            .executables
            .get(&suffix_length)
            .unwrap()
            .get(&prefix_length)
            .unwrap()
    }

    fn compile(
        &self,
        suffix_length: usize,
        prefix_length: usize,
    ) -> Retained<Executable> {
        let config = &self.decoder_config;
        let attention_executable = attention_executable(
            config
                .layer_config
                .attention_config
                .qkv_projection_config
                .activation_precision()
                .into(),
            config.head_dim,
            config.num_groups,
            config.num_heads,
            config.attention_scale,
            &self.mtl_context,
            suffix_length,
            prefix_length,
            &self.compilation_config.descriptor_general,
        );
        attention_executable
    }
}
