use crate::{
    backends::{
        common::{
            AsBufferRangeRef, Buffer, Encoder, Kernels,
            gpu_types::gemm::{gemm_tiling_simdgroups_per_column, gemm_tiling_simdgroups_per_row},
            kernel::attention_gemm::{AttentionGemmArguments, AttentionGemmBackendBlock, GeneratedAttentionGemmBlock},
        },
        metal::{DeviceExt, Metal},
    },
    data_type::DataType,
};

pub mod matmul;

pub const MTLB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/default.metallib"));

include!(concat!(env!("OUT_DIR"), "/dsl.rs"));

pub struct MetalKernels;

impl AttentionGemmBackendBlock for GeneratedAttentionGemmBlock<MetalKernels> {
    type Backend = Metal;

    fn new(data_type: DataType) -> Self {
        GeneratedAttentionGemmBlock::new(data_type)
    }

    fn encode<KVBuf: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &self,
        encoder: &mut Encoder<Metal>,
        args: AttentionGemmArguments<Metal, KVBuf>,
    ) -> Result<(), crate::backends::metal::error::MetalError> {
        let use_accelerator = encoder.context().device.supports_mxu()
            && self.data_type() != DataType::F32
            && args.head_dim <= 128
            && args.suffix_length >= 64
            && std::env::var("UZU_FORCE_SIMD").is_err();
        self.encode_with_accelerator(use_accelerator, encoder, args)
    }
}

impl Kernels for MetalKernels {
    type Backend = Metal;

    autogen_kernels!();
    type AttentionGemmBlock = GeneratedAttentionGemmBlock<MetalKernels>;
    type MatmulKernel = matmul::MatmulMetalKernel;
}
