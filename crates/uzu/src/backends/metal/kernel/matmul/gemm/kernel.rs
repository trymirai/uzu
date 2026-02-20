use std::collections::HashMap;

use metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLFunctionConstantValues};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{DispatchDescriptor, pipeline_configuration::PipelineConfiguration, tile_configuration::TileConfiguration};
use crate::{
    DataType,
    backends::metal::{
        context::MetalContext,
        error::MetalError,
        kernel::matmul::common::{MatmulArguments, transpose_configuration},
        metal_extensions::{ComputeEncoderSetValue, FunctionConstantValuesSetValue},
    },
};

pub struct Kernel {
    data_type: DataType,
    pipelines: HashMap<PipelineConfiguration, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl Kernel {
    pub fn new(data_type: DataType) -> Result<Self, MetalError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MetalError::Generic(format!("Unsupported dtype for GEMM: {data_type:?}")));
        }
        Ok(Self {
            data_type,
            pipelines: HashMap::new(),
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn precompile(
        &mut self,
        context: &MetalContext,
    ) -> Result<(), MetalError> {
        let tiles_and_alignments: &[(TileConfiguration, &[(bool, bool, bool)])] = match self.data_type {
            DataType::BF16 => &[
                (TileConfiguration::new(64, 32, 32, 2, 2, 0), &[(false, true, true), (true, true, true)]),
                (
                    TileConfiguration::new(64, 64, 16, 2, 2, 0),
                    &[(false, true, true), (true, false, true), (true, true, true)],
                ),
                (TileConfiguration::new(64, 64, 16, 1, 2, 0), &[(true, true, true)]),
            ],
            DataType::F16 => {
                &[(TileConfiguration::new(64, 64, 16, 2, 2, 0), &[(true, true, true), (false, true, true)])]
            },
            DataType::F32 => {
                &[(TileConfiguration::new(32, 64, 16, 1, 2, 0), &[(false, true, true), (true, true, true)])]
            },
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
        let transpose_suffix = transpose_configuration(configuration.transpose_a, configuration.transpose_b);
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
        context: &MetalContext,
        configuration: &PipelineConfiguration,
    ) -> Result<&Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        if !self.pipelines.contains_key(configuration) {
            let kernel_name = self.kernel_name(configuration);
            let function_constants = MTLFunctionConstantValues::new();
            function_constants.set_value(&configuration.has_batch, 10);
            function_constants.set_value(&configuration.use_out_source, 100);
            function_constants.set_value(&configuration.do_axpby, 110);
            function_constants.set_value(&configuration.align_m, 200);
            function_constants.set_value(&configuration.align_n, 201);
            function_constants.set_value(&configuration.align_k, 202);

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
            let pipeline_state =
                context.compute_pipeline_state_cached(&cache_key, &kernel_name, Some(&function_constants))?;
            self.pipelines.insert(configuration.clone(), pipeline_state);
        }
        Ok(self.pipelines.get(configuration).unwrap())
    }

    pub(crate) fn encode_descriptor(
        &mut self,
        context: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: &MatmulArguments,
        descriptor: &DispatchDescriptor,
    ) -> Result<bool, MetalError> {
        let pipeline_state = self.get_or_compile_pipeline(context, &descriptor.pipeline_configuration)?;
        encoder.set_compute_pipeline_state(pipeline_state);

        encoder.set_buffer(Some(arguments.a), arguments.a_offset as usize, 0);
        encoder.set_buffer(Some(arguments.b), 0, 1);
        if descriptor.pipeline_configuration.use_out_source {
            if let Some(c_buffer) = arguments.c {
                encoder.set_buffer(Some(c_buffer), 0, 2);
            }
        }
        encoder.set_buffer(Some(arguments.d), 0, 3);

        encoder.set_value(&descriptor.params, 4);

        if let Some(addmm_params) = &descriptor.addmm_params {
            encoder.set_value(addmm_params, 5);
        }

        encoder.dispatch_threadgroups(descriptor.threadgroups, descriptor.threads_per_threadgroup);
        Ok(false)
    }
}
