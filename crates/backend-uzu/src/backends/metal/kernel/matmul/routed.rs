use std::collections::{HashMap, hash_map::Entry};

use crate::{
    backends::{
        common::{
            Buffer, BufferArg, Encoder,
            gpu_types::gemm::GemmDTransform,
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError},
        },
        metal::{
            Metal,
            context::MetalContext,
            kernel::{ExpertRouteCountMetalKernel, ExpertRouteScatterMetalKernel, RoutedGemmMetalKernel},
        },
    },
    data_type::DataType,
};

#[derive(Clone, Copy)]
struct RoutedBufferSlice<'a> {
    buffer: &'a dyn Buffer<Backend = Metal>,
    offset: usize,
    length: usize,
}

impl<'a> RoutedBufferSlice<'a> {
    fn from_arg<T: BufferArg<'a, Metal>>(argument: T) -> Self {
        let (buffer, offset, length) = argument.into_parts();
        Self {
            buffer,
            offset,
            length,
        }
    }
}

impl<'a> BufferArg<'a, Metal> for RoutedBufferSlice<'a> {
    fn into_parts(self) -> (&'a dyn Buffer<Backend = Metal>, usize, usize) {
        (self.buffer, self.offset, self.length)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RoutedGemmSpecialization {
    output_transform: GemmDTransform,
    expert_bias: bool,
    microfloat_group_size: Option<u32>,
    expert_routed: bool,
}

pub(super) struct RoutedGemm {
    count: ExpertRouteCountMetalKernel,
    scatter: ExpertRouteScatterMetalKernel,
    pipelines: HashMap<RoutedGemmSpecialization, RoutedGemmMetalKernel>,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
}

impl RoutedGemm {
    pub(super) fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        Ok(Self {
            count: ExpertRouteCountMetalKernel::new(context).map_err(MatmulError::BackendError)?,
            scatter: ExpertRouteScatterMetalKernel::new(context).map_err(MatmulError::BackendError)?,
            pipelines: HashMap::new(),
            weights_data_type,
            input_data_type,
            output_data_type,
        })
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: RoutedGemmSpecialization,
    ) -> Result<&RoutedGemmMetalKernel, MatmulError<Metal>> {
        match self.pipelines.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let pipeline = RoutedGemmMetalKernel::new(
                    context,
                    self.input_data_type,
                    self.weights_data_type,
                    self.output_data_type,
                    specialization.microfloat_group_size.is_some(),
                    specialization.microfloat_group_size.unwrap_or(0),
                    specialization.expert_routed,
                    specialization.output_transform,
                    specialization.expert_bias,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(pipeline))
            },
        }
    }

    pub(super) fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let output_transform = arguments.d_transform.mask();
        if output_transform.intersects(GemmDTransform::ACCUMULATE | GemmDTransform::RHT) {
            return Err(MatmulError::UnsupportedRouting {
                path: "RoutedGemm",
                reason: "accumulation and RHT are not supported for grouped expert routes",
            });
        }
        if !arguments.b_transpose || arguments.b_leading_dimension.is_some() {
            return Err(MatmulError::UnsupportedRouting {
                path: "RoutedGemm",
                reason: "grouped expert routes require contiguous output-input weights",
            });
        }
        let routes = arguments.routing.expert_routes();
        let MatmulA::FullPrecision {
            values: input,
            offset: input_offset,
        } = arguments.a
        else {
            return Err(MatmulError::IncompatibleA {
                path: "RoutedGemm",
                reason: "prepared int8 activations are not supported",
            });
        };
        let (weights, scales, outer_scales, microfloat_group_size): (
            RoutedBufferSlice<'b>,
            Option<RoutedBufferSlice<'b>>,
            Option<RoutedBufferSlice<'b>>,
            Option<u32>,
        ) = match arguments.b {
            MatmulB::FullPrecision {
                b,
            } => {
                if routes.is_none() {
                    return Err(MatmulError::UnsupportedRouting {
                        path: "RoutedGemm",
                        reason: "dense grouped execution is reserved for microfloat weights",
                    });
                }
                (RoutedBufferSlice::from_arg(b), None, None, None)
            },
            MatmulB::Microfloat {
                codes,
                scales,
                outer_scales,
                metadata,
            } => (
                RoutedBufferSlice::from_arg(codes),
                Some(RoutedBufferSlice::from_arg(scales)),
                Some(RoutedBufferSlice::from_arg(outer_scales)),
                Some(metadata.group_size()),
            ),
            _ => {
                return Err(MatmulError::UnsupportedRouting {
                    path: "RoutedGemm",
                    reason: "grouped expert routes require full-precision or microfloat weights",
                });
            },
        };

        let expert_count = routes.map_or(1, |routes| routes.expert_count.get());
        let route_count = arguments.m;
        let (offsets, grouped_routes) = if let Some(routes) = routes {
            let mut offsets = encoder
                .allocate_scratch_for_shape(&[expert_count + 1], DataType::U32)
                .map_err(MatmulError::BackendError)?;
            let mut cursors = encoder
                .allocate_scratch_for_shape(&[expert_count], DataType::U32)
                .map_err(MatmulError::BackendError)?;
            let mut grouped_routes = encoder
                .allocate_scratch_for_shape(&[route_count], DataType::U32)
                .map_err(MatmulError::BackendError)?;

            self.count.encode(routes.expert_ids, &mut offsets, &mut cursors, route_count, expert_count, encoder);
            self.scatter.encode(routes.expert_ids, &mut cursors, &mut grouped_routes, route_count, expert_count, encoder);
            (Some(offsets), Some(grouped_routes))
        } else {
            (None, None)
        };

        // Invalid expert IDs have no grouped row, so clear their route-major destinations.
        if routes.is_some() {
            encoder.encode_fill(&mut *arguments.d, 0);
        }

        let specialization = RoutedGemmSpecialization {
            output_transform,
            expert_bias: arguments.d_transform.per_matrix_bias.is_some(),
            microfloat_group_size,
            expert_routed: routes.is_some(),
        };
        let pipeline = self.get_or_create(encoder.context(), specialization)?;
        let rows_per_partition = expert_count.saturating_mul(256).max(1);
        let row_partitions = route_count.div_ceil(rows_per_partition).clamp(1, 16);
        pipeline.encode(
            weights,
            scales,
            outer_scales,
            (input, input_offset),
            &mut *arguments.d,
            arguments.d_transform.bias,
            arguments.d_transform.per_matrix_bias,
            offsets.as_ref(),
            grouped_routes.as_ref(),
            route_count,
            arguments.n,
            arguments.k,
            routes.map_or(1, |routes| routes.routes_per_token.get()),
            expert_count,
            routes.is_none_or(|routes| routes.input == crate::backends::common::kernel::matmul::ExpertInput::Routes),
            row_partitions,
            arguments.d_transform.ab_scale,
            arguments.d_transform.soft_cap,
            encoder,
        );
        Ok(())
    }
}
