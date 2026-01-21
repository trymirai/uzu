use std::collections::HashMap;

use crate::backends::metal::{
    ComputeCommandEncoderRef, ComputeEncoderLegacy, ComputePipelineState,
};

use super::{
    dispatch_descriptor::{AxpbySource, DispatchDescriptor},
    pipeline_configuration::PipelineConfiguration,
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError, kernel::matmul::common::MatmulArguments,
    },
};

fn gemv_kernel_name(
    data_type: DataType,
    config: &PipelineConfiguration,
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

    let prefix = if config.transpose_matrix {
        "gemv_t"
    } else {
        "gemv"
    };

    Ok(format!(
        "{prefix}_{dtype_name}_bm{}_bn{}_sm{}_sn{}_tm{}_tn{}_nc{}_axpby{}",
        config.threadgroup_rows,
        config.threadgroup_cols,
        config.threads_per_simdgroup_row,
        config.threads_per_simdgroup_col,
        config.elements_per_thread_row,
        config.elements_per_thread_col,
        config.non_contiguous_batch as u8,
        config.do_axpby as u8,
    ))
}

pub struct Kernel {
    data_type: DataType,
    pipelines: HashMap<PipelineConfiguration, ComputePipelineState>,
}

impl Kernel {
    pub fn new(data_type: DataType) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported data type for GEMV: {:?}",
                data_type
            )));
        }
        Ok(Self {
            data_type,
            pipelines: HashMap::new(),
        })
    }

    pub fn precompile(
        &mut self,
        context: &MTLContext,
    ) -> Result<(), MTLError> {
        let configs: &[(u32, bool)] = match self.data_type {
            DataType::BF16 => &[(4, false), (4, true), (8, false), (8, true)],
            DataType::F16 => &[(8, false)],
            DataType::F32 => &[(8, false)],
            _ => return Ok(()),
        };

        for &(threadgroup_rows, do_axpby) in configs {
            let config = PipelineConfiguration {
                transpose_a: false,
                transpose_b: true,
                transpose_matrix: false,
                threadgroup_rows,
                threadgroup_cols: 1,
                threads_per_simdgroup_row: 1,
                threads_per_simdgroup_col: 32,
                elements_per_thread_row: 4,
                elements_per_thread_col: 4,
                non_contiguous_batch: false,
                do_axpby,
            };
            let _ = self.get_pipeline(context, &config);
        }

        Ok(())
    }

    fn get_pipeline(
        &mut self,
        context: &MTLContext,
        config: &PipelineConfiguration,
    ) -> Result<&ComputePipelineState, MTLError> {
        if !self.pipelines.contains_key(config) {
            let kernel_name = gemv_kernel_name(self.data_type, config)?;
            let pipeline =
                context.compute_pipeline_state(&kernel_name, None)?;
            self.pipelines.insert(*config, pipeline);
        }
        Ok(self.pipelines.get(config).unwrap())
    }

    pub(crate) fn encode_descriptor(
        &mut self,
        context: &MTLContext,
        encoder: ComputeCommandEncoderRef<'_>,
        arguments: &MatmulArguments,
        descriptor: &DispatchDescriptor,
    ) -> Result<bool, MTLError> {
        let pipeline =
            self.get_pipeline(context, &descriptor.pipeline_configuration)?;
        encoder.set_compute_pipeline_state(pipeline);

        let (buf0, off0) = if descriptor.matrix_is_rhs {
            (arguments.b, 0)
        } else {
            (arguments.a, arguments.a_offset)
        };
        encoder.set_buffer(0, Some(buf0), off0);

        let (buf1, off1) = if descriptor.matrix_is_rhs {
            (arguments.a, arguments.a_offset)
        } else {
            (arguments.b, 0)
        };
        encoder.set_buffer(1, Some(buf1), off1);

        if descriptor.pipeline_configuration.do_axpby {
            match descriptor.axpby_source {
                AxpbySource::None => {
                    return Err(MTLError::Generic(
                        "GEMV descriptor mismatch: do_axpby=true but axpby_source=None"
                            .to_owned(),
                    ));
                },
                AxpbySource::Bias => {
                    let bias = arguments.bias.ok_or_else(|| {
                        MTLError::Generic(
                            "GEMV descriptor requires bias buffer".to_owned(),
                        )
                    })?;
                    encoder.set_buffer(2, Some(bias), 0);
                },
                AxpbySource::C => {
                    let c_buffer = arguments.c.ok_or_else(|| {
                        MTLError::Generic(
                            "GEMV descriptor requires C buffer".to_owned(),
                        )
                    })?;
                    encoder.set_buffer(2, Some(c_buffer), 0);
                },
            }
        }

        encoder.set_buffer(3, Some(arguments.d), 0);

        encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &descriptor.input_dimension as *const i32
                as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &descriptor.output_dimension as *const i32
                as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<i32>() as u64,
            &descriptor.matrix_leading_dim as *const i32
                as *const std::ffi::c_void,
        );

        encoder.set_bytes(
            7,
            std::mem::size_of::<f32>() as u64,
            &descriptor.alpha as *const f32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            8,
            std::mem::size_of::<f32>() as u64,
            &descriptor.beta as *const f32 as *const std::ffi::c_void,
        );

        encoder.set_bytes(
            9,
            std::mem::size_of::<i32>() as u64,
            &descriptor.batch_ndim as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            10,
            (std::mem::size_of::<i32>() * descriptor.batch_shape.len()) as u64,
            descriptor.batch_shape.as_ptr() as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            11,
            (std::mem::size_of::<i64>() * descriptor.vector_batch_stride.len())
                as u64,
            descriptor.vector_batch_stride.as_ptr() as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            12,
            (std::mem::size_of::<i64>() * descriptor.matrix_batch_stride.len())
                as u64,
            descriptor.matrix_batch_stride.as_ptr() as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            13,
            (std::mem::size_of::<i64>() * descriptor.bias_batch_stride.len())
                as u64,
            descriptor.bias_batch_stride.as_ptr() as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            14,
            std::mem::size_of::<i32>() as u64,
            &descriptor.bias_stride as *const i32 as *const std::ffi::c_void,
        );

        encoder.dispatch_thread_groups(
            descriptor.threadgroups,
            descriptor.threads_per_threadgroup,
        );

        Ok(descriptor.bias_is_fused())
    }
}
