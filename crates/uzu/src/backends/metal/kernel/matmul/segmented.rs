use std::collections::HashMap;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use super::shared_types::GEMMParams;
use crate::{
    DataType,
    backends::metal::{DeviceClass, MTLContext, MTLError},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SegmentedPipelineKey {
    name: String,
    segments_contiguous: bool,
    align_m: bool,
    align_n: bool,
}

#[derive(Debug, Clone)]
pub struct SegmentedMmArguments<'a> {
    pub a: &'a MTLBuffer,
    pub b: &'a MTLBuffer,
    pub segments: &'a MTLBuffer,
    pub out: &'a MTLBuffer,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    /// Grid Z dimension for the segmented kernel (out.size / (M*N) in the reference backend).
    pub batch_size_out: i32,
    pub segments_contiguous: bool,
    pub transpose_a: bool,
    pub transpose_b: bool,
}

pub struct SegmentedGemm {
    data_type: DataType,
    pipelines: HashMap<SegmentedPipelineKey, MTLComputePipelineState>,
}

impl SegmentedGemm {
    pub fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            pipelines: HashMap::new(),
        }
    }

    fn steel_type_name(&self) -> Result<&'static str, MTLError> {
        match self.data_type {
            DataType::F16 => Ok("float16"),
            DataType::BF16 => Ok("bfloat16"),
            DataType::F32 => Ok("float32"),
            other => Err(MTLError::Generic(format!(
                "Unsupported data type for segmented GEMM: {other:?}"
            ))),
        }
    }

    fn device_class_code(mtl: &MTLContext) -> char {
        match mtl.architecture.device_class {
            DeviceClass::Desktop => 'd',
            DeviceClass::Integrated => 'g',
            DeviceClass::Phone => 'p',
            DeviceClass::Unknown(_) => 'g',
        }
    }

    fn select_gemm_tparams(
        device_class: char,
        data_type: DataType,
        transpose_a: bool,
        transpose_b: bool,
        m: i32,
        n: i32,
        k: i32,
        batch_size_out: i32,
    ) -> (i32, i32, i32, i32, i32) {
        // Matches the initialization and tparam selection logic used by the
        // reference backend for segmented GEMM kernels.
        let mut bm = 64;
        let mut bn = 64;
        let mut bk = 16;
        let mut wm = 2;
        let mut wn = 2;

        let is_float32 = matches!(data_type, DataType::F32);
        let work_elements = (batch_size_out as i64) * (m as i64) * (n as i64);

        match device_class {
            'g' | 'p' => {
                if !transpose_a && transpose_b {
                    bm = 64;
                    bn = 32;
                    bk = 32;
                    wm = 2;
                    wn = 2;
                } else if !is_float32 {
                    bm = 64;
                    bn = 64;
                    bk = 16;
                    wm = 1;
                    wn = 2;
                }
            },
            'd' => {
                if work_elements >= (1_i64 << 20) {
                    if !is_float32 {
                        if 2 * std::cmp::max(m, n) > k {
                            bm = 64;
                            bn = 64;
                            bk = 16;
                            wm = 1;
                            wn = 2;
                        } else if !transpose_a && transpose_b {
                            bm = 64;
                            bn = 32;
                            bk = 32;
                            wm = 2;
                            wn = 2;
                        } else {
                            bm = 32;
                            bn = 64;
                            bk = 16;
                            wm = 1;
                            wn = 2;
                        }
                    }
                } else if !is_float32 {
                    if !transpose_a && transpose_b {
                        bm = 64;
                        bn = 32;
                        bk = 32;
                        wm = 2;
                        wn = 2;
                    } else {
                        bm = 64;
                        bn = 64;
                        bk = 16;
                        wm = 1;
                        wn = 2;
                    }
                }
            },
            _ => {},
        }

        (bm, bn, bk, wm, wn)
    }

    fn segmented_kernel_name(
        &self,
        transpose_a: bool,
        transpose_b: bool,
        bm: i32,
        bn: i32,
        bk: i32,
        wm: i32,
        wn: i32,
    ) -> Result<String, MTLError> {
        let tname = format!(
            "{}{}",
            if transpose_a {
                't'
            } else {
                'n'
            },
            if transpose_b {
                't'
            } else {
                'n'
            }
        );
        let dtype = self.steel_type_name()?;
        Ok(format!(
            "steel_segmented_mm_{tname}_{dtype}_{dtype}_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}"
        ))
    }

    fn get_or_compile_pipeline(
        &mut self,
        mtl: &MTLContext,
        name: &str,
        segments_contiguous: bool,
        align_m: bool,
        align_n: bool,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = SegmentedPipelineKey {
            name: name.to_string(),
            segments_contiguous,
            align_m,
            align_n,
        };

        if !self.pipelines.contains_key(&key) {
            let fcv = metal::FunctionConstantValues::new();
            fcv.set_constant_value_at_index(
                &segments_contiguous as *const bool as *const _,
                metal::MTLDataType::Bool,
                199,
            );
            fcv.set_constant_value_at_index(
                &align_m as *const bool as *const _,
                metal::MTLDataType::Bool,
                200,
            );
            fcv.set_constant_value_at_index(
                &align_n as *const bool as *const _,
                metal::MTLDataType::Bool,
                201,
            );

            let cache_key = format!(
                "{name}_sc{}_am{}_an{}",
                segments_contiguous as u8, align_m as u8, align_n as u8
            );
            let (ps, _) = mtl.compute_pipeline_state_with_reflection_cached(
                &cache_key,
                name,
                Some(&fcv),
            )?;
            self.pipelines.insert(key.clone(), ps);
        }

        Ok(self.pipelines.get(&key).unwrap())
    }

    pub fn encode(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &SegmentedMmArguments,
    ) -> Result<(), MTLError> {
        let device_class = Self::device_class_code(mtl);
        let (bm, bn, bk, wm, wn) = Self::select_gemm_tparams(
            device_class,
            self.data_type,
            args.transpose_a,
            args.transpose_b,
            args.m,
            args.n,
            args.k,
            args.batch_size_out,
        );

        let align_m = (args.m % bm) == 0;
        let align_n = (args.n % bn) == 0;

        let kname = self.segmented_kernel_name(
            args.transpose_a,
            args.transpose_b,
            bm,
            bn,
            bk,
            wm,
            wn,
        )?;
        let ps = self.get_or_compile_pipeline(
            mtl,
            &kname,
            args.segments_contiguous,
            align_m,
            align_n,
        )?;

        enc.set_compute_pipeline_state(ps);
        enc.set_buffer(0, Some(args.a), 0);
        enc.set_buffer(1, Some(args.b), 0);
        enc.set_buffer(2, Some(args.segments), 0);
        enc.set_buffer(3, Some(args.out), 0);

        let tiles_n = (args.n + bn - 1) / bn;
        let tiles_m = (args.m + bm - 1) / bm;
        let params = GEMMParams {
            M: args.m,
            N: args.n,
            K: args.k,
            lda: args.lda,
            ldb: args.ldb,
            ldd: args.n,
            tiles_n,
            tiles_m,
            batch_stride_a: 0,
            batch_stride_b: 0,
            batch_stride_d: (args.m as i64) * (args.n as i64),
            swizzle_log: 0,
            gemm_k_iterations_aligned: 0,
            batch_ndim: 0,
        };
        enc.set_bytes(
            4,
            std::mem::size_of::<GEMMParams>() as u64,
            &params as *const GEMMParams as *const _,
        );

        let threads_per_tg = MTLSize::new(32, wn as u64, wm as u64);
        let tgs = MTLSize::new(
            tiles_n as u64,
            tiles_m as u64,
            args.batch_size_out as u64,
        );
        enc.dispatch_thread_groups(tgs, threads_per_tg);
        Ok(())
    }
}
