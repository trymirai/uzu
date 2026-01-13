use metal::{Buffer as MTLBuffer, ComputeCommandEncoderRef};

use super::{
    super::{KernelDataType, TensorAddBias},
    common::MatmulArguments,
    gemm,
    gemv::GemvKernel,
    split_k::SplitKGemm,
};
use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
    utils::env_utils::debug_matmul_enabled,
};

pub struct MatmulKernel {
    data_type: DataType,
    lhs_is_transposed: bool,
    rhs_is_transposed: bool,
    gemm: Option<gemm::GemmKernel>,
    gemv: Option<GemvKernel>,
    splitk: Option<SplitKGemm>,
    bias_add: Option<TensorAddBias>,
}

impl MatmulKernel {
    pub fn new(
        _mtl: &MTLContext,
        data_type: DataType,
        lhs_is_transposed: bool,
        rhs_is_transposed: bool,
    ) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for MatmulKernel: {data_type:?}"
            )));
        }

        Ok(Self {
            data_type,
            lhs_is_transposed,
            rhs_is_transposed,
            gemm: None,
            gemv: None,
            splitk: None,
            bias_add: None,
        })
    }

    fn maybe_use_gemv_impl(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
        bias: Option<&MTLBuffer>,
    ) -> Result<bool, MTLError> {
        let m = args.batch;
        let n = args.output_dim;
        if self.lhs_is_transposed || !self.rhs_is_transposed {
            return Ok(false);
        }

        if m != 1 && n != 1 {
            return Ok(false);
        }
        let gemv = self.gemv.get_or_insert_with(|| {
            GemvKernel::new(
                self.data_type,
                self.lhs_is_transposed,
                self.rhs_is_transposed,
            )
        });
        if let Some(bias) = bias {
            gemv.encode_with_bias(mtl, enc, args, bias)?;
        } else {
            gemv.encode(mtl, enc, args)?;
        }
        Ok(true)
    }

    fn maybe_use_gemv(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
    ) -> Result<bool, MTLError> {
        self.maybe_use_gemv_impl(mtl, enc, args, None)
    }

    fn maybe_use_gemv_with_bias(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
        bias: &MTLBuffer,
    ) -> Result<bool, MTLError> {
        self.maybe_use_gemv_impl(mtl, enc, args, Some(bias))
    }

    fn maybe_use_splitk(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
    ) -> Result<bool, MTLError> {
        let m = args.batch;
        let n = args.output_dim;
        let k = args.input_dim;
        let batch_count = args.batch_count;

        if !SplitKGemm::should_use_splitk(m, n, k, batch_count) {
            return Ok(false);
        }

        let splitk = self.splitk.get_or_insert_with(|| {
            SplitKGemm::new(
                self.data_type,
                self.lhs_is_transposed,
                self.rhs_is_transposed,
            )
        });
        splitk.encode(mtl, enc, args)?;
        Ok(true)
    }

    fn get_or_create_gemm(
        &mut self
    ) -> Result<&mut gemm::GemmKernel, MTLError> {
        if self.gemm.is_none() {
            self.gemm = Some(gemm::GemmKernel::new(
                self.data_type,
                self.lhs_is_transposed,
                self.rhs_is_transposed,
            )?);
        }
        Ok(self.gemm.as_mut().unwrap())
    }

    fn encode_gemm(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
    ) -> Result<(), MTLError> {
        let gemm = self.get_or_create_gemm()?;
        gemm.encode(mtl, enc, args)
    }

    pub fn encode(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        mut args: MatmulArguments,
    ) -> Result<(), MTLError> {
        self.apply_batch_collapse(&mut args);

        if self.maybe_use_gemv(mtl, enc, &args)? {
            if debug_matmul_enabled() {
                self.log_gemv(&args);
            }
            return Ok(());
        }

        if self.maybe_use_splitk(mtl, enc, &args)? {
            if debug_matmul_enabled() {
                self.log_splitk(&args);
            }
            return Ok(());
        }

        if debug_matmul_enabled() {
            self.log_gemm(&args, mtl);
        }

        self.encode_gemm(mtl, enc, &args)
    }

    pub fn encode_with_bias(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        mut args: MatmulArguments,
        bias: &MTLBuffer,
    ) -> Result<(), MTLError> {
        self.apply_batch_collapse(&mut args);

        if self.maybe_use_gemv_with_bias(mtl, enc, &args, bias)? {
            if debug_matmul_enabled() {
                self.log_gemv(&args);
            }
            return Ok(());
        }

        if self.maybe_use_splitk(mtl, enc, &args)? {
            if debug_matmul_enabled() {
                self.log_splitk(&args);
            }
            self.apply_bias_add(mtl, enc, &args, bias)?;
            return Ok(());
        }

        if debug_matmul_enabled() {
            self.log_gemm(&args, mtl);
        }

        self.encode_gemm(mtl, enc, &args)?;
        self.apply_bias_add(mtl, enc, &args, bias)?;
        Ok(())
    }

    fn apply_bias_add(
        &mut self,
        mtl: &MTLContext,
        enc: &ComputeCommandEncoderRef,
        args: &MatmulArguments,
        bias: &MTLBuffer,
    ) -> Result<(), MTLError> {
        let m = args.batch as usize;
        let n = args.output_dim as usize;
        let batch_count = args.batch_count as usize;
        let total_len = m * n * batch_count;
        if total_len == 0 {
            return Ok(());
        }

        if self.bias_add.is_none() {
            self.bias_add = Some(TensorAddBias::new(
                mtl,
                KernelDataType::from(self.data_type),
            )?);
        }
        let bias_add = self.bias_add.as_ref().unwrap();
        bias_add
            .encode_with_encoder(args.d, bias, args.d, n, total_len, enc, None);
        Ok(())
    }

    fn apply_batch_collapse(
        &self,
        args: &mut MatmulArguments,
    ) {
        if self.lhs_is_transposed {
            return;
        }
        if args.batch_count <= 1 {
            return;
        }
        if args.lda == args.input_dim && self.rhs_is_transposed {
            args.batch *= args.batch_count;
            args.batch_count = 1;
        }
    }

    fn log_gemv(
        &self,
        args: &MatmulArguments,
    ) {
        eprintln!(
            "[matmul] GEMV m={} k={} n={} batch={} dtype={:?}",
            args.batch,
            args.input_dim,
            args.output_dim,
            args.batch_count,
            self.data_type
        );
    }

    fn log_splitk(
        &self,
        args: &MatmulArguments,
    ) {
        eprintln!(
            "[matmul] SplitK m={} k={} n={} batch={} dtype={:?}",
            args.batch,
            args.input_dim,
            args.output_dim,
            args.batch_count,
            self.data_type
        );
    }

    fn log_gemm(
        &self,
        args: &MatmulArguments,
        mtl: &MTLContext,
    ) {
        let gemm_ref = self.gemm.as_ref();
        if let Some(gemm) = gemm_ref {
            let tile = gemm.select_tile(mtl, args);
            eprintln!(
                "[matmul] GEMM m={} k={} n={} batch={} dtype={:?} tile={}x{}x{} nax={}",
                args.batch,
                args.input_dim,
                args.output_dim,
                args.batch_count,
                self.data_type,
                tile.block_rows,
                tile.block_cols,
                tile.block_depth,
                tile.is_nax()
            );
        }
    }
}
