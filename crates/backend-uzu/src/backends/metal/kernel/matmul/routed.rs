use std::collections::{HashMap, hash_map::Entry};

use crate::{
    backends::{
        common::{
            BufferArg, Encoder,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RoutedGemmSpecialization {
    output_transform: GemmDTransform,
    expert_bias: bool,
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
        let routes = arguments.expert_routes.ok_or(MatmulError::UnsupportedRouting {
            path: "RoutedGemm",
            reason: "expert route metadata is required",
        })?;
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
        let MatmulB::FullPrecision {
            b: weights,
        } = arguments.b
        else {
            return Err(MatmulError::UnsupportedRouting {
                path: "RoutedGemm",
                reason: "grouped expert routes currently require full-precision weights",
            });
        };

        let expert_count = routes.expert_count.get();
        let route_count = arguments.m;
        let mut offsets = encoder
            .allocate_scratch_for_shape(&[expert_count + 1], DataType::U32)
            .map_err(MatmulError::BackendError)?;
        let mut cursors =
            encoder.allocate_scratch_for_shape(&[expert_count], DataType::U32).map_err(MatmulError::BackendError)?;
        let mut grouped_routes =
            encoder.allocate_scratch_for_shape(&[route_count], DataType::U32).map_err(MatmulError::BackendError)?;

        self.count.encode(routes.expert_ids, &mut offsets, &mut cursors, route_count, expert_count, encoder);
        self.scatter.encode(routes.expert_ids, &mut cursors, &mut grouped_routes, route_count, expert_count, encoder);
        encoder.encode_fill(&mut *arguments.d, 0);

        let specialization = RoutedGemmSpecialization {
            output_transform,
            expert_bias: routes.expert_biases.is_some(),
        };
        let pipeline = self.get_or_create(encoder.context(), specialization)?;
        let row_partitions = route_count.div_ceil(64).clamp(1, 16);
        pipeline.encode(
            weights,
            (input, input_offset),
            &mut *arguments.d,
            arguments.d_transform.bias,
            routes.expert_biases,
            &offsets,
            &grouped_routes,
            route_count,
            arguments.n,
            arguments.k,
            routes.routes_per_token.get(),
            expert_count,
            routes.input == crate::backends::common::kernel::matmul::ExpertInput::Routes,
            row_partitions,
            arguments.d_transform.ab_scale,
            arguments.d_transform.soft_cap,
            encoder,
        );
        Ok(())
    }
}
