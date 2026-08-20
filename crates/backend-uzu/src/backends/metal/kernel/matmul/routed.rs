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
            command_buffer::{MetalRoutePlan, MetalRoutePlanKey},
            context::MetalContext,
            kernel::{
                ExpertRouteClearCountsMetalKernel, ExpertRouteCountMetalKernel, ExpertRoutePrefixMetalKernel,
                ExpertRouteScatterMetalKernel, ExpertRouteZeroInvalidMetalKernel, RoutedGemmMetalKernel,
            },
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

/// Pipeline identity for dense microfloat and backend-grouped expert GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RowTiledGemmSpecialization {
    output_transform: GemmDTransform,
    expert_bias: bool,
    microfloat_group_size: Option<u32>,
    expert_routed: bool,
}

/// Row-tiled GEMM arm selected after direct routes cross the GEMV threshold.
pub(super) struct RowTiledGemm {
    clear_counts: ExpertRouteClearCountsMetalKernel,
    count: ExpertRouteCountMetalKernel,
    prefix: ExpertRoutePrefixMetalKernel,
    scatter: ExpertRouteScatterMetalKernel,
    zero_invalid: ExpertRouteZeroInvalidMetalKernel,
    pipelines: HashMap<RowTiledGemmSpecialization, RoutedGemmMetalKernel>,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
}

impl RowTiledGemm {
    pub(super) fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        Ok(Self {
            clear_counts: ExpertRouteClearCountsMetalKernel::new(context).map_err(MatmulError::BackendError)?,
            count: ExpertRouteCountMetalKernel::new(context).map_err(MatmulError::BackendError)?,
            prefix: ExpertRoutePrefixMetalKernel::new(context).map_err(MatmulError::BackendError)?,
            scatter: ExpertRouteScatterMetalKernel::new(context).map_err(MatmulError::BackendError)?,
            zero_invalid: ExpertRouteZeroInvalidMetalKernel::new(context, output_data_type)
                .map_err(MatmulError::BackendError)?,
            pipelines: HashMap::new(),
            weights_data_type,
            input_data_type,
            output_data_type,
        })
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: RowTiledGemmSpecialization,
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
                path: "RowTiledGemm",
                reason: "accumulation and RHT are not supported for grouped expert routes",
            });
        }
        if !arguments.b_transpose || arguments.b_leading_dimension.is_some() {
            return Err(MatmulError::UnsupportedRouting {
                path: "RowTiledGemm",
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
                path: "RowTiledGemm",
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
                        path: "RowTiledGemm",
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
                    path: "RowTiledGemm",
                    reason: "grouped expert routes require full-precision or microfloat weights",
                });
            },
        };

        let expert_count = routes.map_or(1, |routes| routes.expert_count.get());
        let route_count = arguments.m;
        let (route_plan_key, route_plan, cache_route_plan) = if let Some(routes) = routes {
            let route_plan_key = MetalRoutePlanKey::new(
                routes.expert_ids,
                route_count,
                expert_count,
                routes.routes_per_token.get(),
            );
            let route_plan = encoder.as_command_buffer_mut().take_route_plan(route_plan_key);
            let (route_plan, cache_route_plan) = match route_plan {
                // W2 consumes the plan produced for W13. Do not reinsert it:
                // the next layer has a distinct TopK allocation identity.
                Some(route_plan) => (route_plan, false),
                None => {
                    let mut offsets = encoder
                        .allocate_scratch_for_shape(&[expert_count + 1], DataType::U32)
                        .map_err(MatmulError::BackendError)?;
                    let mut cursors = encoder
                        .allocate_scratch_for_shape(&[expert_count], DataType::U32)
                        .map_err(MatmulError::BackendError)?;
                    let mut grouped_routes = encoder
                        .allocate_scratch_for_shape(&[route_count], DataType::U32)
                        .map_err(MatmulError::BackendError)?;

                    self.clear_counts.encode(&mut offsets, expert_count, encoder);
                    self.count.encode(routes.expert_ids, &mut offsets, route_count, expert_count, encoder);
                    self.prefix.encode(&mut offsets, &mut cursors, expert_count, encoder);
                    self.scatter.encode(
                        routes.expert_ids,
                        &mut cursors,
                        &mut grouped_routes,
                        route_count,
                        expert_count,
                        encoder,
                    );
                    (
                        MetalRoutePlan {
                            offsets,
                            grouped_routes,
                        },
                        true,
                    )
                },
            };
            (Some(route_plan_key), Some(route_plan), cache_route_plan)
        } else {
            (None, None, false)
        };

        // Valid rows are written by grouped GEMM; zero only exceptional IDs.
        if let Some(routes) = routes {
            self.zero_invalid.encode(
                routes.expert_ids,
                &mut *arguments.d,
                route_count,
                arguments.n,
                expert_count,
                encoder,
            );
        }

        let specialization = RowTiledGemmSpecialization {
            output_transform,
            expert_bias: arguments.d_transform.per_matrix_bias.is_some(),
            microfloat_group_size,
            expert_routed: routes.is_some(),
        };
        let pipeline = self.get_or_create(encoder.context(), specialization)?;
        // Counts stay GPU-private, so partition against the worst-case hot
        // segment. Uniform routes pay only an early return in empty partitions.
        let row_partitions = route_count.div_ceil(256).clamp(1, 16);
        pipeline.encode(
            weights,
            scales,
            outer_scales,
            (input, input_offset),
            &mut *arguments.d,
            arguments.d_transform.bias,
            arguments.d_transform.per_matrix_bias,
            route_plan.as_ref().map(|plan| &plan.offsets),
            route_plan.as_ref().map(|plan| &plan.grouped_routes),
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
        if cache_route_plan && let (Some(route_plan_key), Some(route_plan)) = (route_plan_key, route_plan) {
            encoder.as_command_buffer_mut().insert_route_plan(route_plan_key, route_plan);
        }
        Ok(())
    }
}
