use std::collections::HashMap;

use metal::MTLComputeCommandEncoder;
use objc2::runtime::ProtocolObject;

use super::{DispatchDescriptor, pipeline_configuration::PipelineConfiguration, tile_configuration::TileConfiguration};
use crate::{
    DataType,
    backends::{
        common::kernel::{
            MatmulGemmTile32x64x16Warp1x2Kernel, MatmulGemmTile64x32x32Warp2x2Kernel,
            MatmulGemmTile64x64x16Warp1x2Kernel, MatmulGemmTile64x64x16Warp2x2Kernel,
        },
        metal::{
            MetalContext, MetalError,
            kernel::{
                dsl::{
                    MatmulGemmTile32x64x16Warp1x2MetalKernel, MatmulGemmTile64x32x32Warp2x2MetalKernel,
                    MatmulGemmTile64x64x16Warp1x2MetalKernel, MatmulGemmTile64x64x16Warp2x2MetalKernel,
                },
                matmul::common::MatmulArguments,
            },
        },
    },
};

macro_rules! encode_pipeline {
    ($kernel:expr, $arguments:expr, $descriptor:expr, $group_count_x:expr, $group_count_y:expr, $group_count_z:expr, $encoder:expr) => {
        $kernel.encode(
            ($arguments.a, $arguments.a_offset as usize),
            $arguments.b,
            $arguments.d,
            std::slice::from_ref(&$descriptor.params),
            $group_count_x,
            $group_count_y,
            $group_count_z,
            $encoder,
        );
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GemmShape {
    Tile64x64x16Warp2x2,
    Tile64x64x16Warp1x2,
    Tile64x32x32Warp2x2,
    Tile32x64x16Warp1x2,
}

impl GemmShape {
    fn from_tile(tile: &TileConfiguration) -> Option<Self> {
        if *tile == TileConfiguration::new(64, 64, 16, 2, 2, tile.swizzle_log2) {
            return Some(Self::Tile64x64x16Warp2x2);
        }
        if *tile == TileConfiguration::new(64, 64, 16, 1, 2, tile.swizzle_log2) {
            return Some(Self::Tile64x64x16Warp1x2);
        }
        if *tile == TileConfiguration::new(64, 32, 32, 2, 2, tile.swizzle_log2) {
            return Some(Self::Tile64x32x32Warp2x2);
        }
        if *tile == TileConfiguration::new(32, 64, 16, 1, 2, tile.swizzle_log2) {
            return Some(Self::Tile32x64x16Warp1x2);
        }
        None
    }
}

enum Pipeline {
    Tile64x64x16Warp2x2(MatmulGemmTile64x64x16Warp2x2MetalKernel),
    Tile64x64x16Warp1x2(MatmulGemmTile64x64x16Warp1x2MetalKernel),
    Tile64x32x32Warp2x2(MatmulGemmTile64x32x32Warp2x2MetalKernel),
    Tile32x64x16Warp1x2(MatmulGemmTile32x64x16Warp1x2MetalKernel),
}

pub struct Kernel {
    data_type: DataType,
    pipelines: HashMap<PipelineConfiguration, Pipeline>,
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

    pub fn precompile(
        &mut self,
        context: &MetalContext,
    ) -> Result<(), MetalError> {
        let precompile_tiles_and_alignments: &[(TileConfiguration, &[(bool, bool, bool)])] = match self.data_type {
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
            _ => &[],
        };

        for (tile, alignments) in precompile_tiles_and_alignments {
            for &(align_m, align_n, align_k) in *alignments {
                let configuration = PipelineConfiguration {
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
                let _ = self.get_or_create_pipeline(context, &configuration)?;
            }
        }

        Ok(())
    }

    fn supports_configuration(configuration: &PipelineConfiguration) -> bool {
        !configuration.transpose_a
            && configuration.transpose_b
            && !configuration.has_batch
            && !configuration.use_out_source
            && !configuration.do_axpby
            && !configuration.tile.is_nax()
    }

    fn get_or_create_pipeline(
        &mut self,
        context: &MetalContext,
        configuration: &PipelineConfiguration,
    ) -> Result<&Pipeline, MetalError> {
        if !self.pipelines.contains_key(configuration) {
            let shape = GemmShape::from_tile(&configuration.tile).ok_or_else(|| {
                MetalError::Generic(format!("Unsupported GEMM tile: {:?}", configuration.tile))
            })?;
            let pipeline = match shape {
                GemmShape::Tile64x64x16Warp2x2 => {
                    Pipeline::Tile64x64x16Warp2x2(MatmulGemmTile64x64x16Warp2x2MetalKernel::new(
                        context,
                        self.data_type,
                        configuration.align_m,
                        configuration.align_n,
                        configuration.align_k,
                    )?)
                },
                GemmShape::Tile64x64x16Warp1x2 => {
                    Pipeline::Tile64x64x16Warp1x2(MatmulGemmTile64x64x16Warp1x2MetalKernel::new(
                        context,
                        self.data_type,
                        configuration.align_m,
                        configuration.align_n,
                        configuration.align_k,
                    )?)
                },
                GemmShape::Tile64x32x32Warp2x2 => {
                    Pipeline::Tile64x32x32Warp2x2(MatmulGemmTile64x32x32Warp2x2MetalKernel::new(
                        context,
                        self.data_type,
                        configuration.align_m,
                        configuration.align_n,
                        configuration.align_k,
                    )?)
                },
                GemmShape::Tile32x64x16Warp1x2 => {
                    Pipeline::Tile32x64x16Warp1x2(MatmulGemmTile32x64x16Warp1x2MetalKernel::new(
                        context,
                        self.data_type,
                        configuration.align_m,
                        configuration.align_n,
                        configuration.align_k,
                    )?)
                },
            };
            self.pipelines.insert(configuration.clone(), pipeline);
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
        let configuration = &descriptor.pipeline_configuration;
        if !Self::supports_configuration(configuration) {
            return Err(MetalError::Generic(format!(
                "Unsupported GEMM configuration: {configuration:?}"
            )));
        }

        let group_count_x = u32::try_from(descriptor.threadgroups.width).map_err(|_| {
            MetalError::Generic(format!(
                "GEMM group count x overflows u32: {}",
                descriptor.threadgroups.width
            ))
        })?;
        let group_count_y = u32::try_from(descriptor.threadgroups.height).map_err(|_| {
            MetalError::Generic(format!(
                "GEMM group count y overflows u32: {}",
                descriptor.threadgroups.height
            ))
        })?;
        let group_count_z = u32::try_from(descriptor.threadgroups.depth).map_err(|_| {
            MetalError::Generic(format!(
                "GEMM group count z overflows u32: {}",
                descriptor.threadgroups.depth
            ))
        })?;

        let pipeline = self.get_or_create_pipeline(context, configuration)?;
        match pipeline {
            Pipeline::Tile64x64x16Warp2x2(kernel) => {
                encode_pipeline!(
                    kernel,
                    arguments,
                    descriptor,
                    group_count_x,
                    group_count_y,
                    group_count_z,
                    encoder
                );
            },
            Pipeline::Tile64x64x16Warp1x2(kernel) => {
                encode_pipeline!(
                    kernel,
                    arguments,
                    descriptor,
                    group_count_x,
                    group_count_y,
                    group_count_z,
                    encoder
                );
            },
            Pipeline::Tile64x32x32Warp2x2(kernel) => {
                encode_pipeline!(
                    kernel,
                    arguments,
                    descriptor,
                    group_count_x,
                    group_count_y,
                    group_count_z,
                    encoder
                );
            },
            Pipeline::Tile32x64x16Warp1x2(kernel) => {
                encode_pipeline!(
                    kernel,
                    arguments,
                    descriptor,
                    group_count_x,
                    group_count_y,
                    group_count_z,
                    encoder
                );
            },
        }

        Ok(false)
    }
}
