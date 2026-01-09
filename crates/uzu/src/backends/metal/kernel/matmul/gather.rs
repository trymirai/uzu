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
struct RhsOnlyPipelineKey {
    name: String,
    align_m: bool,
    align_n: bool,
    align_k: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GatherPipelineKey {
    name: String,
    has_batch: bool,
    align_m: bool,
    align_n: bool,
    align_k: bool,
}

#[derive(Debug, Clone)]
pub struct GatherMmRhsArguments<'a> {
    pub a: &'a MTLBuffer,
    pub b: &'a MTLBuffer,
    pub rhs_indices: &'a MTLBuffer,
    pub out: &'a MTLBuffer,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    pub batch_stride_b: i64,
    pub transpose_b: bool,
    /// Number of matrices available on the RHS (E dimension in the reference backend).
    pub rhs_matrix_count: i32,
}

#[derive(Debug, Clone)]
pub struct GatherMmArguments<'a> {
    pub a: &'a MTLBuffer,
    pub b: &'a MTLBuffer,
    pub lhs_indices: &'a MTLBuffer,
    pub rhs_indices: &'a MTLBuffer,
    pub out: &'a MTLBuffer,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    /// Grid Z dimension for the gather kernel (out.size / (M*N) in the reference backend).
    pub batch_size_out: i32,
    /// Number of batch dimensions for the output (out.ndim - 2 in the reference backend).
    pub batch_ndim: i32,

    // LHS/RHS indices layout (used when has_batch=true).
    pub lhs_indices_shape: &'a [i32],
    pub lhs_indices_strides: &'a [i64],
    pub rhs_indices_strides: &'a [i64],

    // A tensor batch layout (for elem_to_loc on selected matrix index).
    pub batch_ndim_a: i32,
    pub a_shape: &'a [i32],
    pub a_strides: &'a [i64],

    // B tensor batch layout (for elem_to_loc on selected matrix index).
    pub batch_ndim_b: i32,
    pub b_shape: &'a [i32],
    pub b_strides: &'a [i64],

    pub transpose_a: bool,
    pub transpose_b: bool,
}

pub struct GatherGemm {
    data_type: DataType,
    rhs_only_pipelines: HashMap<RhsOnlyPipelineKey, MTLComputePipelineState>,
    gather_pipelines: HashMap<GatherPipelineKey, MTLComputePipelineState>,
}

impl GatherGemm {
    pub fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            rhs_only_pipelines: HashMap::new(),
            gather_pipelines: HashMap::new(),
        }
    }

    fn steel_type_name(&self) -> Result<&'static str, MTLError> {
        match self.data_type {
            DataType::F16 => Ok("float16"),
            DataType::BF16 => Ok("bfloat16"),
            DataType::F32 => Ok("float32"),
            other => Err(MTLError::Generic(format!(
                "Unsupported data type for gather GEMM: {other:?}"
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
        // reference backend for gather/segmented GEMM kernels.
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
                    // nt
                    bm = 64;
                    bn = 32;
                    bk = 32;
                    wm = 2;
                    wn = 2;
                } else if !is_float32 {
                    // half and bfloat
                    bm = 64;
                    bn = 64;
                    bk = 16;
                    wm = 1;
                    wn = 2;
                }
            },
            'd' => {
                if work_elements >= (1_i64 << 20) {
                    // large matmul
                    if !is_float32 {
                        if 2 * std::cmp::max(m, n) > k {
                            // Reasonable K
                            bm = 64;
                            bn = 64;
                            bk = 16;
                            wm = 1;
                            wn = 2;
                        } else if !transpose_a && transpose_b {
                            // nt with large k
                            bm = 64;
                            bn = 32;
                            bk = 32;
                            wm = 2;
                            wn = 2;
                        } else {
                            // nn with large K
                            bm = 32;
                            bn = 64;
                            bk = 16;
                            wm = 1;
                            wn = 2;
                        }
                    }
                } else if !is_float32 {
                    // smaller matmul, half and bfloat
                    if !transpose_a && transpose_b {
                        // nt
                        bm = 64;
                        bn = 32;
                        bk = 32;
                        wm = 2;
                        wn = 2;
                    } else {
                        // nn
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

    fn gather_kernel_name(
        &self,
        prefix: &str,
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
            "{prefix}_{tname}_{dtype}_{dtype}_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}",
        ))
    }

    fn set_i32_slice_bytes(
        enc: &ComputeCommandEncoderRef,
        index: u64,
        data: &[i32],
    ) {
        if data.is_empty() {
            let zero = 0_i32;
            enc.set_bytes(
                index,
                std::mem::size_of::<i32>() as u64,
                &zero as *const i32 as *const _,
            );
        } else {
            enc.set_bytes(
                index,
                (data.len() * std::mem::size_of::<i32>()) as u64,
                data.as_ptr() as *const _,
            );
        }
    }

    fn set_i64_slice_bytes(
        enc: &ComputeCommandEncoderRef,
        index: u64,
        data: &[i64],
    ) {
        if data.is_empty() {
            let zero = 0_i64;
            enc.set_bytes(
                index,
                std::mem::size_of::<i64>() as u64,
                &zero as *const i64 as *const _,
            );
        } else {
            enc.set_bytes(
                index,
                (data.len() * std::mem::size_of::<i64>()) as u64,
                data.as_ptr() as *const _,
            );
        }
    }

    fn get_or_compile_rhs_only_pipeline(
        &mut self,
        mtl: &MTLContext,
        name: &str,
        align_m: bool,
        align_n: bool,
        align_k: bool,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = RhsOnlyPipelineKey {
            name: name.to_string(),
            align_m,
            align_n,
            align_k,
        };

        if !self.rhs_only_pipelines.contains_key(&key) {
            let fcv = metal::FunctionConstantValues::new();
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
            fcv.set_constant_value_at_index(
                &align_k as *const bool as *const _,
                metal::MTLDataType::Bool,
                202,
            );

            let cache_key = format!(
                "{name}_am{}_an{}_ak{}",
                align_m as u8, align_n as u8, align_k as u8
            );
            let (ps, _) = mtl.compute_pipeline_state_with_reflection_cached(
                &cache_key,
                name,
                Some(&fcv),
            )?;
            self.rhs_only_pipelines.insert(key.clone(), ps);
        }

        Ok(self.rhs_only_pipelines.get(&key).unwrap())
    }

    fn get_or_compile_gather_pipeline(
        &mut self,
        mtl: &MTLContext,
        name: &str,
        has_batch: bool,
        align_m: bool,
        align_n: bool,
        align_k: bool,
    ) -> Result<&MTLComputePipelineState, MTLError> {
        let key = GatherPipelineKey {
            name: name.to_string(),
            has_batch,
            align_m,
            align_n,
            align_k,
        };

        if !self.gather_pipelines.contains_key(&key) {
            let fcv = metal::FunctionConstantValues::new();
            fcv.set_constant_value_at_index(
                &has_batch as *const bool as *const _,
                metal::MTLDataType::Bool,
                10,
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
            fcv.set_constant_value_at_index(
                &align_k as *const bool as *const _,
                metal::MTLDataType::Bool,
                202,
            );

            let cache_key = format!(
                "{name}_hb{}_am{}_an{}_ak{}",
                has_batch as u8, align_m as u8, align_n as u8, align_k as u8
            );
            let (ps, _) = mtl.compute_pipeline_state_with_reflection_cached(
                &cache_key,
                name,
                Some(&fcv),
            )?;
            self.gather_pipelines.insert(key.clone(), ps);
        }

        Ok(self.gather_pipelines.get(&key).unwrap())
    }

    pub fn encode_mm_rhs(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &GatherMmRhsArguments,
    ) -> Result<(), MTLError> {
        // NAX gather kernels are only instantiated for float16/bfloat16.
        let prefer_nax =
            mtl.is_nax_available() && !matches!(self.data_type, DataType::F32);

        let (bm, bn, bk, wm, wn, use_nax) = if prefer_nax {
            // Mirrors rhs-only NAX gather tile selection.
            let e = std::cmp::max(1, args.rhs_matrix_count);
            let ratio = args.m / e;
            let (bm, wm) = if ratio > 48 {
                (64, 2)
            } else if ratio > 24 {
                (32, 1)
            } else {
                (16, 1)
            };
            (bm, 128, 128, wm, 4, true)
        } else {
            (16, 64, 16, 1, 2, false)
        };

        let align_m = (args.m % bm) == 0;
        let align_n = (args.n % bn) == 0;
        let align_k = (args.k % bk) == 0;

        let prefix = if use_nax {
            "steel_gather_mm_rhs_nax"
        } else {
            "steel_gather_mm_rhs"
        };

        // gather_mm_rhs only supports transpose_a=false (nn/nt instantiations).
        let kname = self.gather_kernel_name(
            prefix,
            /*transpose_a=*/ false,
            args.transpose_b,
            bm,
            bn,
            bk,
            wm,
            wn,
        )?;

        let ps = self.get_or_compile_rhs_only_pipeline(
            mtl, &kname, align_m, align_n, align_k,
        )?;

        enc.set_compute_pipeline_state(ps);
        enc.set_buffer(0, Some(args.a), 0);
        enc.set_buffer(1, Some(args.b), 0);
        enc.set_buffer(2, Some(args.rhs_indices), 0);
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
            batch_stride_b: args.batch_stride_b,
            batch_stride_d: 0,
            swizzle_log: 0,
            gemm_k_iterations_aligned: args.k / bk,
            batch_ndim: 0,
        };
        enc.set_bytes(
            4,
            std::mem::size_of::<GEMMParams>() as u64,
            &params as *const GEMMParams as *const _,
        );

        let threads_per_tg = MTLSize::new(32, wn as u64, wm as u64);
        let tgs = MTLSize::new(tiles_n as u64, tiles_m as u64, 1);
        enc.dispatch_thread_groups(tgs, threads_per_tg);
        Ok(())
    }

    pub fn encode_mm(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &GatherMmArguments,
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

        let has_batch = args.batch_ndim > 1;
        let align_m = (args.m % bm) == 0;
        let align_n = (args.n % bn) == 0;
        let align_k = (args.k % bk) == 0;

        let kname = self.gather_kernel_name(
            "steel_gather_mm",
            args.transpose_a,
            args.transpose_b,
            bm,
            bn,
            bk,
            wm,
            wn,
        )?;

        let ps = self.get_or_compile_gather_pipeline(
            mtl, &kname, has_batch, align_m, align_n, align_k,
        )?;

        enc.set_compute_pipeline_state(ps);

        enc.set_buffer(0, Some(args.a), 0);
        enc.set_buffer(1, Some(args.b), 0);
        enc.set_buffer(2, Some(args.lhs_indices), 0);
        enc.set_buffer(3, Some(args.rhs_indices), 0);
        enc.set_buffer(4, Some(args.out), 0);

        let tiles_n = (args.n + bn - 1) / bn;
        let tiles_m = (args.m + bm - 1) / bm;
        let batch_stride_a = if args.batch_ndim > 0 {
            *args.lhs_indices_strides.first().unwrap_or(&0)
        } else {
            0
        };
        let batch_stride_b = if args.batch_ndim > 0 {
            *args.rhs_indices_strides.first().unwrap_or(&0)
        } else {
            0
        };

        let params = GEMMParams {
            M: args.m,
            N: args.n,
            K: args.k,
            lda: args.lda,
            ldb: args.ldb,
            ldd: args.n,
            tiles_n,
            tiles_m,
            batch_stride_a,
            batch_stride_b,
            batch_stride_d: (args.m as i64) * (args.n as i64),
            swizzle_log: 0,
            gemm_k_iterations_aligned: args.k / bk,
            batch_ndim: args.batch_ndim,
        };

        enc.set_bytes(
            5,
            std::mem::size_of::<GEMMParams>() as u64,
            &params as *const GEMMParams as *const _,
        );

        // Note: many of these are only used when has_batch=true, but they are
        // present in the kernel signature unconditionally.
        Self::set_i32_slice_bytes(enc, 6, args.lhs_indices_shape);
        Self::set_i64_slice_bytes(enc, 7, args.lhs_indices_strides);
        Self::set_i64_slice_bytes(enc, 8, args.rhs_indices_strides);

        enc.set_bytes(
            9,
            std::mem::size_of::<i32>() as u64,
            &args.batch_ndim_a as *const i32 as *const _,
        );
        Self::set_i32_slice_bytes(enc, 10, args.a_shape);
        Self::set_i64_slice_bytes(enc, 11, args.a_strides);

        enc.set_bytes(
            12,
            std::mem::size_of::<i32>() as u64,
            &args.batch_ndim_b as *const i32 as *const _,
        );
        Self::set_i32_slice_bytes(enc, 13, args.b_shape);
        Self::set_i64_slice_bytes(enc, 14, args.b_strides);

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
