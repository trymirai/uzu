use std::collections::HashMap;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use super::split_k;
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError,
        kernel::{matmul::common::GEMMParams, mlp::MlpActivationType},
    },
};

#[derive(Debug)]
pub struct Arguments<'a> {
    pub input: &'a MTLBuffer,
    pub input_offset: u64,
    pub weights: &'a MTLBuffer,
    pub output: &'a MTLBuffer,
    pub batch: i32,
    pub input_dim: i32,
    pub hidden_dim: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldd: i32,
    pub activation: MlpActivationType,
}

pub struct Kernel {
    data_type: DataType,
    weights_transposed: bool,
    pipelines: HashMap<
        (i32, i32, bool, bool, MlpActivationType),
        MTLComputePipelineState,
    >,
    splitk_kernel: Option<split_k::Kernel>,
}

impl Kernel {
    pub fn new(
        data_type: DataType,
        weights_transposed: bool,
    ) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for MLP fused GEMM: {:?}",
                data_type
            )));
        }
        let splitk_kernel = if weights_transposed
            && matches!(data_type, DataType::F16 | DataType::BF16)
        {
            Some(split_k::Kernel::new(data_type))
        } else {
            None
        };
        Ok(Self {
            data_type,
            weights_transposed,
            pipelines: HashMap::new(),
            splitk_kernel,
        })
    }

    fn kernel_name(
        &self,
        bm: i32,
        bn: i32,
        mn_aligned: bool,
        k_aligned: bool,
    ) -> String {
        let type_name = match self.data_type {
            DataType::F16 => "half",
            DataType::BF16 => "bfloat",
            DataType::F32 => "float",
            _ => unreachable!(),
        };
        let transpose_char = if self.weights_transposed {
            "t"
        } else {
            "n"
        };
        let mn_str = if mn_aligned {
            "true"
        } else {
            "false"
        };
        let k_str = if k_aligned {
            "true"
        } else {
            "false"
        };

        format!(
            "steel_gemm_mlp_fused_{}_{}_{}_bm{}_bn{}_bk16_wm2_wn2_align_MN_{}_K_{}",
            transpose_char, type_name, type_name, bm, bn, mn_str, k_str
        )
    }

    fn get_pipeline(
        &mut self,
        context: &MTLContext,
        bm: i32,
        bn: i32,
        mn_aligned: bool,
        k_aligned: bool,
        activation: MlpActivationType,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = (bm, bn, mn_aligned, k_aligned, activation);
        if !self.pipelines.contains_key(&key) {
            let name = self.kernel_name(bm, bn, mn_aligned, k_aligned);

            let fcv = metal::FunctionConstantValues::new();
            let activation_val = activation as u32;
            fcv.set_constant_value_at_index(
                &activation_val as *const u32 as *const _,
                metal::MTLDataType::UInt,
                52,
            );

            let pipeline = context.compute_pipeline_state(&name, Some(&fcv))?;
            self.pipelines.insert(key, pipeline);
        }
        Ok(self.pipelines.get(&key).unwrap())
    }

    fn select_tile_size(
        &self,
        m: i32,
        hidden_dim: i32,
        _k: i32,
    ) -> (i32, i32) {
        if m >= 64 && hidden_dim >= 64 {
            (64, 64)
        } else {
            (32, 32)
        }
    }

    pub fn encode(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        args: &Arguments,
    ) -> Result<(), MTLError> {
        if let Some(ref mut splitk) = self.splitk_kernel {
            if split_k::Kernel::should_use_splitk(
                args.batch,
                args.hidden_dim,
                args.input_dim,
            ) {
                let splitk_args = split_k::Arguments {
                    input: args.input,
                    input_offset: args.input_offset,
                    weights: args.weights,
                    output: args.output,
                    batch: args.batch,
                    input_dim: args.input_dim,
                    hidden_dim: args.hidden_dim,
                    lda: args.lda,
                    ldb: args.ldb,
                    ldd: args.ldd,
                    activation: args.activation,
                };
                return splitk.encode(context, encoder, &splitk_args);
            }
        }

        let (bm, bn) =
            self.select_tile_size(args.batch, args.hidden_dim, args.input_dim);
        let bk = 16;

        let mn_aligned = args.batch % bm == 0 && args.hidden_dim % bn == 0;
        let k_aligned = args.input_dim % bk == 0;

        let pipeline = self.get_pipeline(
            context,
            bm,
            bn,
            mn_aligned,
            k_aligned,
            args.activation,
        )?;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(args.input), args.input_offset);
        encoder.set_buffer(1, Some(args.weights), 0);
        encoder.set_buffer(2, Some(args.output), 0);

        let tiles_m = (args.batch + bm - 1) / bm;
        let tiles_n = (args.hidden_dim + bn - 1) / bn;
        let gemm_k_iterations = (args.input_dim + bk - 1) / bk;

        let params = GEMMParams {
            M: args.batch,
            N: args.hidden_dim * 2,
            K: args.input_dim,
            lda: args.lda,
            ldb: args.ldb,
            ldd: args.ldd,
            tiles_n,
            tiles_m,
            batch_stride_a: 0,
            batch_stride_b: 0,
            batch_stride_d: 0,
            swizzle_log: 0,
            gemm_k_iterations_aligned: if k_aligned {
                gemm_k_iterations
            } else {
                gemm_k_iterations - 1
            },
            batch_ndim: 0,
        };

        encoder.set_bytes(
            3,
            std::mem::size_of::<GEMMParams>() as u64,
            &params as *const GEMMParams as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            10,
            std::mem::size_of::<i32>() as u64,
            &args.hidden_dim as *const i32 as *const std::ffi::c_void,
        );

        let threadgroup_count = MTLSize::new(tiles_n as u64, tiles_m as u64, 1);
        let threads_per_threadgroup = MTLSize::new(32, 2, 2);

        encoder
            .dispatch_thread_groups(threadgroup_count, threads_per_threadgroup);
        Ok(())
    }
}
