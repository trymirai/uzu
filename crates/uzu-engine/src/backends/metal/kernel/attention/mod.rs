use crate::backends::common::Backend;

mod error;
mod fallback;
mod gemm;
mod gemm_grouped;
mod int8;
mod single_pass;
mod two_pass;

pub use error::AttentionMetalError;

use crate::backends::{
    common::{
        BufferRef, CommandBufferEncoding,
        kernel::{
            AttentionArguments, AttentionKernel, AttentionKernelConfig, Int8AttentionArguments, SigmoidGateKernel,
        },
    },
    metal::{Metal, command_buffer::MetalCommandBufferEncoding, context::MetalContext, kernel::SigmoidGateMetalKernel},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MaskKind {
    None,
    Causal,
    Trie,
}

impl MaskKind {
    fn for_attention(
        is_causal: bool,
        is_trie: bool,
    ) -> Option<Self> {
        match (is_causal, is_trie) {
            (true, true) => Some(Self::Trie),
            (true, false) => Some(Self::Causal),
            (false, false) => Some(Self::None),
            (false, true) => None,
        }
    }

    fn is_causal(self) -> bool {
        !matches!(self, Self::None)
    }

    fn is_trie(self) -> bool {
        matches!(self, Self::Trie)
    }
}

const SINGLE_PASS_KV_THRESHOLD: u32 = 1_024;
const D256_SINGLE_PASS_KV_THRESHOLD: u32 = 150_000;
const D128_SHORT_KV_GEMM_THRESHOLD: u32 = 2_048;

pub struct AttentionMetalKernel {
    config: AttentionKernelConfig,
    gate: Option<(SigmoidGateMetalKernel, u32)>,
    grouped: Option<gemm_grouped::AttentionGemmGrouped>,
    int8: Option<int8::Int8Attention>,
    gemm: Option<gemm::AttentionGemm>,
    fallback: Option<fallback::AttentionFallback>,
    two_pass: two_pass::AttentionTwoPass,
    single_pass: single_pass::AttentionSinglePass,
}

impl AttentionKernel for AttentionMetalKernel {
    type Backend = Metal;
    type Error = AttentionMetalError;

    fn new(
        context: &MetalContext,
        config: AttentionKernelConfig,
    ) -> Result<Self, AttentionMetalError> {
        let grouped = gemm_grouped::AttentionGemmGrouped::is_supported(&config, context)
            .then(|| gemm_grouped::AttentionGemmGrouped::new(&config));
        let int8 = int8::Int8Attention::is_supported(&config, context)
            .then(|| int8::Int8Attention::new(context, &config))
            .transpose()
            .map_err(AttentionMetalError::Backend)?;
        let gemm = gemm::AttentionGemm::is_supported(&config).then(|| gemm::AttentionGemm::new(&config));
        let fallback = fallback::AttentionFallback::is_supported(&config)
            .then(|| fallback::AttentionFallback::new(&config, context))
            .transpose()?;
        let gate = config
            .gate_row_stride
            .map(|stride| SigmoidGateMetalKernel::new(context, config.data_type).map(|kernel| (kernel, stride)))
            .transpose()?;
        Ok(Self {
            config,
            gate,
            grouped,
            int8,
            gemm,
            fallback,
            two_pass: two_pass::AttentionTwoPass::new(&config),
            single_pass: single_pass::AttentionSinglePass::new(&config),
        })
    }

    fn supports_int8_kv(&self) -> bool {
        self.int8.is_some()
    }

    fn encode_int8(
        &self,
        arguments: Int8AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, AttentionMetalError> {
        let command_buffer = &mut *command_buffer.debug_group("attention core");
        self.int8
            .as_ref()
            .expect("INT8 KV unsupported for this attention config")
            .encode(arguments, command_buffer)
            .map_err(AttentionMetalError::Backend)
    }

    fn encode(
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
    ) -> Result<<Metal as Backend>::ScratchBuffer, AttentionMetalError> {
        let command_buffer = &mut *command_buffer.debug_group("attention core");
        let (gate, suffix_length) = (arguments.gate, arguments.suffix_length);
        self.encode_impl(arguments, command_buffer).map(|(mut output, gate_applied)| {
            if let (Some(gate), Some((kernel, stride)), false) = (gate, &self.gate, gate_applied) {
                kernel.encode(
                    gate,
                    &mut output,
                    self.config.num_q_heads * self.config.head_dim,
                    suffix_length,
                    *stride,
                    command_buffer,
                );
            }
            output
        })
    }
}

impl AttentionMetalKernel {
    fn encode_impl(
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
    ) -> Result<(<Metal as Backend>::ScratchBuffer, bool), AttentionMetalError> {
        let suffix_length = arguments.suffix_length;
        let gate_applied_by_two_pass = arguments.gate.is_some();
        let kv_length = arguments.cache.prefix_len() + suffix_length;
        let is_trie = arguments.trie.is_some();
        let is_ring = arguments.cache.ring_params().is_some();
        let mask = MaskKind::for_attention(self.config.is_causal, is_trie);

        if !is_ring
            && let Some(mask) = mask
            && let Some(grouped) = &self.grouped
        {
            if grouped.should_encode(mask, suffix_length, kv_length) {
                return grouped.encode(mask, arguments, command_buffer).map_err(AttentionMetalError::Backend);
            }

            // These measured sibling exceptions apply only when the
            // grouped core was built for this layer.
            if self.config.head_dim == 256
                && (1..=16).contains(&suffix_length)
                && kv_length > D256_SINGLE_PASS_KV_THRESHOLD
            {
                return Ok((self.single_pass.encode(arguments, command_buffer)?, false));
            }
            if (self.config.head_dim == 256 && (9..=16).contains(&suffix_length))
                || (self.config.head_dim == 128
                    && (9..=16).contains(&suffix_length)
                    && kv_length > D128_SHORT_KV_GEMM_THRESHOLD)
            {
                return Ok((self.two_pass.encode(arguments, command_buffer)?, gate_applied_by_two_pass));
            }
        }

        if suffix_length > 8 {
            if let Some(gemm) = &self.gemm {
                return Ok((gemm.encode(arguments, command_buffer)?, false));
            }
            if !is_trie && let Some(fallback) = &self.fallback {
                return Ok((fallback.encode(arguments, command_buffer)?, false));
            }
        }

        if kv_length > SINGLE_PASS_KV_THRESHOLD {
            Ok((self.two_pass.encode(arguments, command_buffer)?, gate_applied_by_two_pass))
        } else {
            Ok((self.single_pass.encode(arguments, command_buffer)?, false))
        }
    }
}

#[cfg(test)]
#[path = "../../../../../unit/backends/metal/kernel/attention/mod.rs"]
mod tests;
