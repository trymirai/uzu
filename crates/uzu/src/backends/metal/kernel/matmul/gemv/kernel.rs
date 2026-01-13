use std::collections::HashMap;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef, ComputePipelineState,
    MTLSize,
};

use super::configuration::{Configuration, select_configuration};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError, kernel::matmul::common::MatmulArguments,
    },
};

fn gemv_kernel_name(
    data_type: DataType,
    configuration: &Configuration,
) -> Result<String, MTLError> {
    let dtype_name = match data_type {
        DataType::F16 => "float16",
        DataType::BF16 => "bfloat16",
        DataType::F32 => "float32",
        _ => {
            return Err(MTLError::Generic(format!(
                "Unsupported data type for GEMV: {:?}",
                data_type
            )));
        },
    };

    let prefix = if configuration.transpose_matrix {
        "gemv_t"
    } else {
        "gemv"
    };

    Ok(format!(
        "{prefix}_{dtype_name}_bm{}_bn{}_sm{}_sn{}_tm{}_tn{}_nc{}_axpby{}",
        configuration.threadgroup_rows,
        configuration.threadgroup_cols,
        configuration.threads_per_simdgroup_row,
        configuration.threads_per_simdgroup_col,
        configuration.elements_per_thread_row,
        configuration.elements_per_thread_col,
        configuration.non_contiguous_batch as u8,
        configuration.do_axpby as u8,
    ))
}

pub struct Kernel {
    data_type: DataType,
    lhs_is_transposed: bool,
    rhs_is_transposed: bool,
    pipelines: HashMap<Configuration, ComputePipelineState>,
}

impl Kernel {
    pub fn new(
        data_type: DataType,
        lhs_is_transposed: bool,
        rhs_is_transposed: bool,
    ) -> Self {
        Self {
            data_type,
            lhs_is_transposed,
            rhs_is_transposed,
            pipelines: HashMap::new(),
        }
    }

    fn get_pipeline(
        &mut self,
        context: &MTLContext,
        configuration: Configuration,
    ) -> Result<&ComputePipelineState, MTLError> {
        if !self.pipelines.contains_key(&configuration) {
            let kernel_name = gemv_kernel_name(self.data_type, &configuration)?;
            let pipeline =
                context.compute_pipeline_state(&kernel_name, None)?;
            self.pipelines.insert(configuration, pipeline);
        }
        Ok(self.pipelines.get(&configuration).unwrap())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_with_configuration(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        arguments: &MatmulArguments,
        configuration: Configuration,
        matrix_is_rhs: bool,
        bias_buffer: Option<&MTLBuffer>,
        alpha: f32,
        beta: f32,
        bias_stride: i32,
    ) -> Result<(), MTLError> {
        let pipeline = self.get_pipeline(context, configuration)?;
        encoder.set_compute_pipeline_state(pipeline);

        let (buf0, off0) = if matrix_is_rhs {
            (arguments.b, 0)
        } else {
            (arguments.a, arguments.a_offset)
        };
        encoder.set_buffer(0, Some(buf0), off0);

        let (buf1, off1) = if matrix_is_rhs {
            (arguments.a, arguments.a_offset)
        } else {
            (arguments.b, 0)
        };
        encoder.set_buffer(1, Some(buf1), off1);

        if configuration.do_axpby {
            if let Some(bias) = bias_buffer {
                encoder.set_buffer(2, Some(bias), 0);
            }
        }

        encoder.set_buffer(3, Some(arguments.d), 0);

        let input_dimension = arguments.input_dim;
        let output_dimension = if matrix_is_rhs {
            arguments.output_dim
        } else {
            arguments.batch
        };
        let matrix_leading_dim = if matrix_is_rhs {
            arguments.ldb
        } else {
            arguments.lda
        };

        encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &input_dimension as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &output_dimension as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<i32>() as u64,
            &matrix_leading_dim as *const i32 as *const std::ffi::c_void,
        );

        encoder.set_bytes(
            7,
            std::mem::size_of::<f32>() as u64,
            &alpha as *const f32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            8,
            std::mem::size_of::<f32>() as u64,
            &beta as *const f32 as *const std::ffi::c_void,
        );

        let batch_shape = if arguments.batch_count > 1 {
            vec![arguments.batch_count]
        } else {
            vec![1]
        };
        let batch_ndim = 1i32;
        let batch_groups = arguments.batch_count.max(1);

        let elements_per_matrix_a =
            (arguments.batch as i64) * (arguments.lda as i64);
        let elements_per_matrix_b = if self.rhs_is_transposed {
            (arguments.output_dim as i64) * (arguments.ldb as i64)
        } else {
            (arguments.input_dim as i64) * (arguments.ldb as i64)
        };

        let vector_batch_stride = if matrix_is_rhs {
            vec![elements_per_matrix_a]
        } else {
            vec![elements_per_matrix_b]
        };

        let matrix_batch_stride = if matrix_is_rhs {
            vec![elements_per_matrix_b]
        } else {
            vec![elements_per_matrix_a]
        };

        let bias_batch_stride = if arguments.batch_count > 1 {
            vec![(output_dimension as i64) * (arguments.ldd as i64)]
        } else {
            vec![0]
        };

        encoder.set_bytes(
            9,
            std::mem::size_of::<i32>() as u64,
            &batch_ndim as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            10,
            (std::mem::size_of::<i32>() * batch_shape.len()) as u64,
            batch_shape.as_ptr() as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            11,
            (std::mem::size_of::<i64>() * vector_batch_stride.len()) as u64,
            vector_batch_stride.as_ptr() as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            12,
            (std::mem::size_of::<i64>() * matrix_batch_stride.len()) as u64,
            matrix_batch_stride.as_ptr() as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            13,
            (std::mem::size_of::<i64>() * bias_batch_stride.len()) as u64,
            bias_batch_stride.as_ptr() as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            14,
            std::mem::size_of::<i32>() as u64,
            &bias_stride as *const i32 as *const std::ffi::c_void,
        );

        let output_elements_per_threadgroup =
            configuration.output_elements_per_threadgroup();
        let threadgroup_count_x =
            ((output_dimension as u32 + output_elements_per_threadgroup - 1)
                / output_elements_per_threadgroup) as u64;
        let threadgroup_count_z = batch_groups.max(1) as u64;

        let threadgroup_count =
            MTLSize::new(threadgroup_count_x, 1, threadgroup_count_z);
        let threads_per_threadgroup = configuration.threads_per_threadgroup();

        encoder
            .dispatch_thread_groups(threadgroup_count, threads_per_threadgroup);
        Ok(())
    }

    pub fn encode(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        arguments: &MatmulArguments,
    ) -> Result<(), MTLError> {
        self.encode_internal(context, encoder, arguments, None)
    }

    pub fn encode_with_bias(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        arguments: &MatmulArguments,
        bias: &MTLBuffer,
    ) -> Result<(), MTLError> {
        self.encode_internal(context, encoder, arguments, Some(bias))
    }

    fn encode_internal(
        &mut self,
        context: &MTLContext,
        encoder: &ComputeCommandEncoderRef,
        arguments: &MatmulArguments,
        bias: Option<&MTLBuffer>,
    ) -> Result<(), MTLError> {
        let m = arguments.batch;
        let n = arguments.output_dim;
        if m != 1 && n != 1 {
            return Ok(());
        }

        let matrix_is_rhs = n != 1;
        let transpose_matrix = if matrix_is_rhs {
            !self.rhs_is_transposed
        } else {
            self.lhs_is_transposed
        };

        let has_non_contiguous_batch = false;

        let (do_axpby, bias_buffer, alpha, beta, bias_stride) =
            if let Some(bias_buf) = bias {
                (true, Some(bias_buf), 1.0f32, 1.0f32, 1)
            } else if let Some(c_buf) = arguments.c {
                (
                    true,
                    Some(c_buf),
                    arguments.alpha,
                    arguments.beta,
                    arguments.ldd,
                )
            } else {
                (false, None, 1.0f32, 0.0f32, 0)
            };

        let configuration = select_configuration(
            transpose_matrix,
            arguments.input_dim,
            if matrix_is_rhs {
                arguments.output_dim
            } else {
                arguments.batch
            },
            has_non_contiguous_batch,
            do_axpby,
        );

        if crate::utils::env_utils::debug_matmul_enabled() {
            let kernel_name = gemv_kernel_name(self.data_type, &configuration)
                .unwrap_or_default();
            eprintln!(
                "[matmul] GEMV m={} k={} n={} batch={} dtype={:?} transpose={} kernel={}",
                arguments.batch,
                arguments.input_dim,
                arguments.output_dim,
                arguments.batch_count,
                self.data_type,
                configuration.transpose_matrix,
                kernel_name
            );
        }

        self.encode_with_configuration(
            context,
            encoder,
            arguments,
            configuration,
            matrix_is_rhs,
            bias_buffer,
            alpha,
            beta,
            bias_stride,
        )
    }
}
