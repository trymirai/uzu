use std::{collections::HashMap, ffi::c_void, ptr::NonNull};

use metal::MTLComputeCommandEncoder;

use super::{
    DispatchDescriptor, pipeline_configuration::PipelineConfiguration,
    tile_configuration::TileConfiguration,
};
use crate::{
    DataType,
    backends::metal::{
        MTLComputePipelineState, MTLContext, MTLError, MTLFunctionConstantValues,
        ProtocolObject, Retained,
        kernel::matmul::common::{
            GEMMAddMMParams, GEMMParams, MatmulArguments,
            transpose_configuration,
        },
    },
};

pub struct Kernel {
    data_type: DataType,
    pipelines: HashMap<PipelineConfiguration, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl Kernel {
    pub fn new(data_type: DataType) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for GEMM: {data_type:?}"
            )));
        }
        Ok(Self {
            data_type,
            pipelines: HashMap::new(),
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn precompile(
        &mut self,
        context: &MTLContext,
    ) -> Result<(), MTLError> {
        let tiles_and_alignments: &[(
            TileConfiguration,
            &[(bool, bool, bool)],
        )] = match self.data_type {
            DataType::BF16 => &[
                (
                    TileConfiguration::new(64, 32, 32, 2, 2, 0),
                    &[(false, true, true), (true, true, true)],
                ),
                (
                    TileConfiguration::new(64, 64, 16, 2, 2, 0),
                    &[
                        (false, true, true),
                        (true, false, true),
                        (true, true, true),
                    ],
                ),
                (
                    TileConfiguration::new(64, 64, 16, 1, 2, 0),
                    &[(true, true, true)],
                ),
            ],
            DataType::F16 => &[(
                TileConfiguration::new(64, 64, 16, 2, 2, 0),
                &[(true, true, true), (false, true, true)],
            )],
            DataType::F32 => &[(
                TileConfiguration::new(32, 64, 16, 1, 2, 0),
                &[(false, true, true), (true, true, true)],
            )],
            _ => return Ok(()),
        };

        for (tile, alignments) in tiles_and_alignments {
            for &(align_m, align_n, align_k) in *alignments {
                let config = PipelineConfiguration {
                    tile: *tile,
                    transpose_a: false,
                    transpose_b: true,
                    align_m,
                    align_n,
                    align_k,
                    has_batch: false,
                    use_out_source: false,
                    do_axpby: false,
                };
                let _ = self.get_or_compile_pipeline(context, &config);
            }
        }

        Ok(())
    }

    fn type_name(&self) -> &'static str {
        match self.data_type {
            DataType::F16 => "float16",
            DataType::BF16 => "bfloat16",
            DataType::F32 => "float32",
            _ => unreachable!(),
        }
    }

    fn kernel_name(
        &self,
        configuration: &PipelineConfiguration,
    ) -> String {
        let type_name = self.type_name();
        let transpose_suffix = transpose_configuration(
            configuration.transpose_a,
            configuration.transpose_b,
        );
        let prefix = if configuration.tile.is_nax() {
            "steel_gemm_nax"
        } else {
            "steel_gemm"
        };
        format!(
            "{}_{}_{}_{}_bm{}_bn{}_bk{}_wm{}_wn{}",
            prefix,
            transpose_suffix.as_str(),
            type_name,
            type_name,
            configuration.tile.block_rows,
            configuration.tile.block_cols,
            configuration.tile.block_depth,
            configuration.tile.warps_per_row,
            configuration.tile.warps_per_col
        )
    }

    fn get_or_compile_pipeline(
        &mut self,
        context: &MTLContext,
        configuration: &PipelineConfiguration,
    ) -> Result<&Retained<ProtocolObject<dyn MTLComputePipelineState>>, MTLError> {
        if !self.pipelines.contains_key(configuration) {
            let kernel_name = self.kernel_name(configuration);
            let function_constants = MTLFunctionConstantValues::new();
            function_constants.set_constant_value_type_at_index(
                NonNull::from(&configuration.has_batch).cast(),
                metal::MTLDataType::Bool,
                10,
            );
            function_constants.set_constant_value_type_at_index(
                NonNull::from(&configuration.use_out_source).cast(),
                metal::MTLDataType::Bool,
                100,
            );
            function_constants.set_constant_value_type_at_index(
                NonNull::from(&configuration.do_axpby).cast(),
                metal::MTLDataType::Bool,
                110,
            );
            function_constants.set_constant_value_type_at_index(
                NonNull::from(&configuration.align_m).cast(),
                metal::MTLDataType::Bool,
                200,
            );
            function_constants.set_constant_value_type_at_index(
                NonNull::from(&configuration.align_n).cast(),
                metal::MTLDataType::Bool,
                201,
            );
            function_constants.set_constant_value_type_at_index(
                NonNull::from(&configuration.align_k).cast(),
                metal::MTLDataType::Bool,
                202,
            );

            let cache_key = format!(
                "{}_am{}_an{}_ak{}_hb{}_uo{}_ax{}",
                kernel_name,
                configuration.align_m as u8,
                configuration.align_n as u8,
                configuration.align_k as u8,
                configuration.has_batch as u8,
                configuration.use_out_source as u8,
                configuration.do_axpby as u8
            );
            let (pipeline_state, _) = context
                .compute_pipeline_state_with_reflection_cached(
                    &cache_key,
                    &kernel_name,
                    Some(&function_constants),
                )?;
            self.pipelines.insert(configuration.clone(), pipeline_state);
        }
        Ok(self.pipelines.get(configuration).unwrap())
    }

    pub(crate) fn encode_descriptor(
        &mut self,
        context: &MTLContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: &MatmulArguments,
        descriptor: &DispatchDescriptor,
    ) -> Result<bool, MTLError> {
        let pipeline_state = self.get_or_compile_pipeline(
            context,
            &descriptor.pipeline_configuration,
        )?;
        encoder.set_compute_pipeline_state(pipeline_state);

        encoder.set_buffer(Some(arguments.a), arguments.a_offset as usize, 0);
        encoder.set_buffer(Some(arguments.b), 0, 1);
        if descriptor.pipeline_configuration.use_out_source {
            if let Some(c_buffer) = arguments.c {
                encoder.set_buffer(Some(c_buffer), 0, 2);
            }
        }
        encoder.set_buffer(Some(arguments.d), 0, 3);

        unsafe {
            encoder.set_bytes(
                NonNull::new(&descriptor.params as *const _ as *mut c_void)
                    .unwrap(),
                std::mem::size_of::<GEMMParams>(),
                4,
            );
        }

        if let Some(addmm_params) = &descriptor.addmm_params {
            unsafe {
                encoder.set_bytes(
                    NonNull::new(addmm_params as *const _ as *mut c_void)
                        .unwrap(),
                    std::mem::size_of::<GEMMAddMMParams>(),
                    5,
                );
            }
        }

        encoder.dispatch_threadgroups(
            descriptor.threadgroups,
            descriptor.threads_per_threadgroup,
        );
        Ok(false)
    }
}
