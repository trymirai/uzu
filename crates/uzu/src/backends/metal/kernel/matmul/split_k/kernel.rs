use std::collections::HashMap;

use crate::backends::metal::{
    Buffer, ComputeCommandEncoderRef, ComputeEncoderLegacy,
    ComputePipelineState as ComputePipelineState, MTLDeviceExt, MTLResourceOptions,
};

use super::{
    DispatchDescriptor, pipeline_configuration::PipelineConfiguration,
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError,
        kernel::matmul::common::{
            GEMMSpiltKParams as SplitKGEMMParams, MatmulArguments,
            transpose_configuration,
        },
    },
};

pub struct Kernel {
    data_type: DataType,
    partial_pipelines: HashMap<PipelineConfiguration, ComputePipelineState>,
    accum_pipeline: Option<ComputePipelineState>,
    accumulator_buffer: Option<Buffer>,
    accumulator_buffer_bytes: usize,
}

impl Kernel {
    pub fn new(data_type: DataType) -> Result<Self, MTLError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)
        {
            return Err(MTLError::Generic(format!(
                "Unsupported dtype for Split-K: {:?}",
                data_type
            )));
        }
        Ok(Self {
            data_type,
            partial_pipelines: HashMap::new(),
            accum_pipeline: None,
            accumulator_buffer: None,
            accumulator_buffer_bytes: 0,
        })
    }

    pub fn precompile(
        &mut self,
        context: &MTLContext,
    ) -> Result<(), MTLError> {
        use super::tile_configuration::TileConfiguration;

        if !matches!(self.data_type, DataType::BF16) {
            return Ok(());
        }

        let tile_configs = [
            TileConfiguration {
                tile_rows: 16,
                tile_cols: 16,
                tile_depth: 16,
                warps_per_row: 2,
                warps_per_col: 2,
            },
            TileConfiguration {
                tile_rows: 16,
                tile_cols: 32,
                tile_depth: 16,
                warps_per_row: 2,
                warps_per_col: 2,
            },
            TileConfiguration {
                tile_rows: 32,
                tile_cols: 16,
                tile_depth: 16,
                warps_per_row: 2,
                warps_per_col: 2,
            },
            TileConfiguration {
                tile_rows: 32,
                tile_cols: 32,
                tile_depth: 16,
                warps_per_row: 2,
                warps_per_col: 2,
            },
        ];

        for tile in &tile_configs {
            for &mn_aligned in &[false, true] {
                let config = PipelineConfiguration {
                    tile: *tile,
                    transpose_a: false,
                    transpose_b: true,
                    mn_aligned,
                    k_aligned: true,
                };
                let _ = self.get_partial_pipeline(context, &config);
            }
        }

        let _ = self.get_accum_pipeline(context);

        Ok(())
    }

    fn steel_type_name(&self) -> Result<&'static str, MTLError> {
        match self.data_type {
            DataType::F16 => Ok("float16"),
            DataType::BF16 => Ok("bfloat16"),
            DataType::F32 => Ok("float32"),
            _ => Err(MTLError::Generic(format!(
                "Unsupported dtype for Split-K: {:?}",
                self.data_type
            ))),
        }
    }

    fn splitk_partial_out_name(&self) -> &'static str {
        "float32"
    }

    fn partial_kernel_name(
        &self,
        config: &PipelineConfiguration,
    ) -> Result<String, MTLError> {
        let in_name = self.steel_type_name()?;
        let out_name = self.splitk_partial_out_name();
        let transpose_suffix =
            transpose_configuration(config.transpose_a, config.transpose_b)
                .as_str();
        let mn_tag = if config.mn_aligned {
            "taligned"
        } else {
            "naligned"
        };
        let k_tag = if config.k_aligned {
            "taligned"
        } else {
            "naligned"
        };
        Ok(format!(
            "steel_gemm_splitk_{}_{}_{}_bm{}_bn{}_bk{}_wm{}_wn{}_MN_{}_K_{}",
            transpose_suffix,
            in_name,
            out_name,
            config.tile.tile_rows,
            config.tile.tile_cols,
            config.tile.tile_depth,
            config.tile.warps_per_row,
            config.tile.warps_per_col,
            mn_tag,
            k_tag,
        ))
    }

    fn accum_kernel_name(&self) -> Result<String, MTLError> {
        let out_name = self.steel_type_name()?;
        Ok(format!(
            "steel_gemm_splitk_accum_{}_{}",
            out_name,
            self.splitk_partial_out_name()
        ))
    }

    fn get_partial_pipeline(
        &mut self,
        mtl: &MTLContext,
        config: &PipelineConfiguration,
    ) -> Result<&ComputePipelineState, MTLError> {
        if !self.partial_pipelines.contains_key(config) {
            let name = self.partial_kernel_name(config)?;
            let ps = mtl.compute_pipeline_state(&name, None)?;
            self.partial_pipelines.insert(config.clone(), ps);
        }
        Ok(self.partial_pipelines.get(config).unwrap())
    }

    fn get_accum_pipeline(
        &mut self,
        mtl: &MTLContext,
    ) -> Result<&ComputePipelineState, MTLError> {
        if self.accum_pipeline.is_none() {
            let name = self.accum_kernel_name()?;
            let ps = mtl.compute_pipeline_state(&name, None)?;
            self.accum_pipeline = Some(ps);
        }
        Ok(self.accum_pipeline.as_ref().unwrap())
    }

    pub fn should_use_splitk(
        m: i32,
        n: i32,
        k: i32,
        batch_count: i32,
    ) -> bool {
        if batch_count != 1 {
            return false;
        }
        if m <= 0 || n <= 0 || k <= 0 {
            return false;
        }
        let m_tiles = m / 16;
        let n_tiles = n / 16;
        let k_tiles = k / 16;
        (m_tiles * n_tiles) <= 32 && k_tiles >= 8
    }

    pub(crate) fn encode_descriptor(
        &mut self,
        context: &MTLContext,
        encoder: ComputeCommandEncoderRef<'_>,
        arguments: &MatmulArguments,
        descriptor: &DispatchDescriptor,
    ) -> Result<bool, MTLError> {
        self.ensure_accumulator_buffer(context, descriptor.accumulator_bytes);
        let accumulator_buffer = self
            .accumulator_buffer
            .as_ref()
            .cloned()
            .expect("Accumulator buffer must be initialized");

        let partial_pipeline_state = self.get_partial_pipeline(
            context,
            &descriptor.pipeline_configuration,
        )?;

        encoder.set_compute_pipeline_state(partial_pipeline_state);
        encoder.set_buffer(0, Some(arguments.a), arguments.a_offset);
        encoder.set_buffer(1, Some(arguments.b), 0);
        encoder.set_buffer(2, Some(&accumulator_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<SplitKGEMMParams>() as u64,
            &descriptor.params as *const _ as *const _,
        );
        encoder.dispatch_thread_groups(
            descriptor.partial_threadgroups,
            descriptor.partial_threads_per_threadgroup,
        );

        let accum_pipeline_state = self.get_accum_pipeline(context)?;
        encoder.set_compute_pipeline_state(accum_pipeline_state);
        encoder.set_buffer(0, Some(&accumulator_buffer), 0);
        encoder.set_buffer(1, Some(arguments.d), 0);

        let partition_count = descriptor.partition_count;
        let output_elements_per_partition =
            descriptor.output_elements_per_partition;
        encoder.set_bytes(
            2,
            std::mem::size_of::<i32>() as u64,
            &partition_count as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &output_elements_per_partition as *const i32
                as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &(arguments.ldd) as *const i32 as *const std::ffi::c_void,
        );

        encoder.dispatch_threads(
            descriptor.accum_total_threads,
            descriptor.accum_threads_per_threadgroup,
        );

        Ok(false)
    }

    fn ensure_accumulator_buffer(
        &mut self,
        mtl: &MTLContext,
        required_bytes: usize,
    ) {
        if required_bytes <= self.accumulator_buffer_bytes
            && self.accumulator_buffer.is_some()
        {
            return;
        }
        self.accumulator_buffer = Some(mtl.device.new_buffer(
            required_bytes as usize,
            MTLResourceOptions::STORAGE_MODE_PRIVATE,
        ).expect("Failed to create accumulator buffer"));
        self.accumulator_buffer_bytes = required_bytes;
    }
}
