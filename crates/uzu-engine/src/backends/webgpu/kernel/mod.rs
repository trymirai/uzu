//! Slang kernels implementing the engine's existing kernel contracts.
use std::collections::BTreeMap;

use super::{
    WebGpu,
    command_buffer::{Binding, WebGpuCommandBufferEncoding},
    error::WebGpuError,
};
use crate::{
    backends::common::{BufferMut, BufferRef, Kernels, kernel::Unsupported},
    data_type::DataType,
};

mod attention;
mod elementwise;
mod embedding;
mod gdn;
mod matmul;
mod normalization;
mod sampling;
mod shader;
mod short_conv;
mod transform;
use shader::{NamedBinding, Param, Shader};

include!(concat!(env!("OUT_DIR"), "/webgpu.rs"));
include!(concat!(env!("OUT_DIR"), "/webgpu_slang.rs"));

pub struct WebGpuKernels;
impl Kernels for WebGpuKernels {
    type Backend = WebGpu;
    webgpu_kernels!();
    type AttentionKernel = attention::WebGpuAttention;
    type MatmulKernel = matmul::WebGpuMatmul;
    type DeltaNetChunkedPrefill = Unsupported<WebGpu>;
    type DeltaNetTreeVerify = Unsupported<WebGpu>;
    type RadixTopKSmall = Unsupported<WebGpu>;
}

struct Kernel {
    name: &'static str,
    params: BTreeMap<&'static str, Param>,
}
impl Kernel {
    fn new(
        name: &'static str,
        params: Vec<(&'static str, Param)>,
        context: &super::context::WebGpuContext,
    ) -> Result<Self, WebGpuError> {
        let mut params: BTreeMap<_, _> = params.into_iter().collect();
        // A fixed subgroup width enables wave-specialized attention and reductions.
        // Unknown or variable widths keep the portable workgroup implementations.
        let subgroup_width = context
            .subgroup_size_range()
            .filter(|&(min, max)| {
                context.supports_subgroups()
                    && context.subgroup_id_supported()
                    && min == max
                    && matches!(min, 32 | 64 | 128)
            })
            .map_or(0, |(width, _)| width);
        params.insert("__subgroup_width", Param::u32(subgroup_width));
        let kernel = Self {
            name,
            params,
        };
        if let Some(dim) = kernel.params.get("HEAD_K_DIM")
            && !(1..=256).contains(&dim.words[0])
        {
            return Err(WebGpuError::Unsupported(format!("WebGPU {name} head dimension {}", dim.words[0])));
        }
        for (key, value) in &kernel.params {
            if let Some(dtype) = value.data_type
                && (!matches!(dtype, DataType::F32 | DataType::F16 | DataType::BF16)
                    || (*key == "AccumT" && dtype != DataType::F32))
            {
                return Err(WebGpuError::Unsupported(format!("WebGPU {name} {key}={dtype:?}")));
            }
        }
        // These variants require separate algorithms. Reject at model setup,
        // rather than accepting a configuration that cannot execute correctly.
        if name == "AttentionSinglePass"
            && (kernel.flag("is_trie") || !(1..=256).contains(&kernel.params["HEAD_DIM"].words[0]))
        {
            return Err(WebGpuError::Unsupported("WebGPU attention variant".into()));
        }
        if (name == "ActivationTransform" && kernel.params["ops"].words[0] >= 2)
            || (name == "GatedActMul" && kernel.params["ops"].words[0] != 0)
            || (name == "InputEmbeddingLookup" && kernel.params["table_kind"].words[0] == 2)
        {
            return Err(WebGpuError::Unsupported(format!("WebGPU {name} quantization variant")));
        }
        Ok(kernel)
    }
    fn flag(
        &self,
        name: &str,
    ) -> bool {
        self.params.get(name).is_some_and(|p| p.words[0] != 0)
    }
    fn data_type(
        &self,
        name: &str,
    ) -> DataType {
        self.params[name].data_type.expect("kernel type parameter")
    }
    fn encode(
        &self,
        bindings: Vec<NamedBinding>,
        params: Vec<(&'static str, Param)>,
        command: &mut WebGpuCommandBufferEncoding,
    ) {
        let mut shader = Shader::new(bindings, self.params.clone(), params);
        match self.name {
            "DeltaNetConvUpdate"
            | "DeltaNetConvScan"
            | "DeltaNetNormGate"
            | "DeltaNetPrefillPrep"
            | "DeltaNetPrefill"
            | "DeltaNetUpdate"
            | "Conv1dPack"
            | "ConvTreeScan"
            | "StateAdvance" => gdn::encode(self.name, &mut shader),
            "Normalization" | "QKVNorm" => normalization::encode(self.name, &mut shader),
            "InputEmbeddingLookup" => embedding::encode(&mut shader),
            "AttentionPrepare" | "AttentionSinglePass" | "KVCacheUpdate" => attention::encode(self.name, &mut shader),
            "UnifiedSampling" | "RepetitionPenalty" | "ContextRingUpdate" => {
                sampling::dispatch(self.name, shader, command);
                return;
            },
            "ShortConvPack" | "ShortConvPrefill" | "ShortConvDecode" | "ShortConvTrie" => {
                short_conv::encode(self.name, &mut shader)
            },
            "ActivationTransform" | "GatedActMul" => transform::encode(self.name, &mut shader),
            _ => elementwise::encode(self.name, &mut shader),
        }
        shader.dispatch(command);
    }
}
