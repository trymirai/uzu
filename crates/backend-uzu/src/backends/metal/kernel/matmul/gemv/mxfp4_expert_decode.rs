use std::{
    collections::{HashMap, hash_map::Entry},
    sync::OnceLock,
};

use crate::{
    backends::{
        common::{
            BufferArg, Encoder,
            gpu_types::gemm::GemmDTransform,
            kernel::matmul::{GateActMulDOps, MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulShape},
            microfloat::MicrofloatLayout,
        },
        metal::{Metal, context::MetalContext, kernel::Mxfp4ExpertDecodeGemvMetalKernel},
    },
    data_type::DataType,
};

/// Decode-sized routed MXFP4 projections run on the dedicated block-native
/// kernel; larger route counts stay on the generic path.
pub(crate) const MXFP4_EXPERT_DECODE_MAX_ROUTES: u32 = 8;

const DEFAULT_DECODE_ROWS_PER_SIMDGROUP: u32 = 4;
const DEFAULT_DECODE_SIMDGROUPS: u32 = 4;

fn decode_geometry() -> (u32, u32) {
    static GEOMETRY: OnceLock<(u32, u32)> = OnceLock::new();
    *GEOMETRY.get_or_init(|| {
        // TODO: remove magic env vars after the geometry sweep settles
        let rows = std::env::var("UZU_MXFP4_DECODE_ROWS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|rows| matches!(rows, 1 | 2 | 4))
            .unwrap_or(DEFAULT_DECODE_ROWS_PER_SIMDGROUP);
        let simdgroups = std::env::var("UZU_MXFP4_DECODE_SIMDGROUPS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|simdgroups| matches!(simdgroups, 2 | 4))
            .unwrap_or(DEFAULT_DECODE_SIMDGROUPS);
        (rows, simdgroups)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Mxfp4ExpertDecodeGemvSpec {
    group_size: u32,
    rows_per_simdgroup: u32,
    num_simdgroups: u32,
    expert_bias: bool,
    custom_activation_alpha: bool,
    clip_gate: bool,
    clip_value: bool,
    fused_gate_up: bool,
}

impl Mxfp4ExpertDecodeGemvSpec {
    pub(crate) fn select(
        shape: &MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        ab_scale: f32,
        gate_act: Option<GateActMulDOps>,
    ) -> Option<Self> {
        let metadata = shape.b_microfloat?;
        if !shape.expert_routed
            || shape.sparse_readout
            || !shape.b_transpose
            || shape.b_leading_dimension.is_some()
            || !shape.a_full_precision
        {
            return None;
        }
        if metadata.layout() != MicrofloatLayout::OutputInput || !matches!(metadata.group_size(), 16 | 32) {
            return None;
        }
        if metadata.rows() != shape.n || metadata.columns() != shape.k {
            return None;
        }
        if shape.m == 0 || shape.m > MXFP4_EXPERT_DECODE_MAX_ROUTES || !shape.k.is_multiple_of(32) {
            return None;
        }
        let expected_transform = if gate_act.is_some() {
            GemmDTransform::GATE_ACT_MUL
        } else {
            GemmDTransform::empty()
        };
        if shape.d_transform != expected_transform || ab_scale != 1.0 {
            return None;
        }
        if let Some(gate_act) = gate_act {
            // The fused epilogue pairs value row h with gate row N/2 + h and
            // writes F32 hidden rows; only the production W13 combination is
            // instantiated.
            if !shape.n.is_multiple_of(2) || input_data_type != DataType::BF16 || output_data_type != DataType::F32 {
                return None;
            }
            let (rows_per_simdgroup, num_simdgroups) = decode_geometry();
            return Some(Self {
                group_size: metadata.group_size(),
                rows_per_simdgroup: rows_per_simdgroup.max(2),
                num_simdgroups,
                expert_bias: shape.expert_bias,
                custom_activation_alpha: gate_act.activation_alpha.is_some(),
                clip_gate: gate_act.gate_clipping.is_some(),
                clip_value: gate_act.value_clipping.is_some(),
                fused_gate_up: true,
            });
        }
        if !matches!(output_data_type, DataType::F32 | DataType::BF16) {
            return None;
        }
        if !matches!(input_data_type, DataType::BF16 | DataType::F32)
            || !matches!(weights_data_type, DataType::BF16 | DataType::F32)
        {
            return None;
        }
        let (rows_per_simdgroup, num_simdgroups) = decode_geometry();
        Some(Self {
            group_size: metadata.group_size(),
            rows_per_simdgroup,
            num_simdgroups,
            expert_bias: shape.expert_bias,
            custom_activation_alpha: false,
            clip_gate: false,
            clip_value: false,
            fused_gate_up: false,
        })
    }

    fn outputs_per_threadgroup(&self) -> u32 {
        let per_simdgroup = if self.fused_gate_up {
            self.rows_per_simdgroup / 2
        } else {
            self.rows_per_simdgroup
        };
        self.num_simdgroups * per_simdgroup
    }
}

pub(crate) struct Mxfp4ExpertDecodeGemvDispatch {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    pipelines: HashMap<Mxfp4ExpertDecodeGemvSpec, Mxfp4ExpertDecodeGemvMetalKernel>,
}

impl Mxfp4ExpertDecodeGemvDispatch {
    pub(crate) fn new(
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Self {
        Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            pipelines: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        spec: Mxfp4ExpertDecodeGemvSpec,
    ) -> Result<&Mxfp4ExpertDecodeGemvMetalKernel, MatmulError<Metal>> {
        match self.pipelines.entry(spec) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = Mxfp4ExpertDecodeGemvMetalKernel::new(
                    context,
                    self.input_data_type,
                    self.weights_data_type,
                    self.output_data_type,
                    spec.group_size,
                    spec.rows_per_simdgroup,
                    spec.num_simdgroups,
                    spec.fused_gate_up,
                    spec.expert_bias,
                    spec.custom_activation_alpha,
                    spec.clip_gate,
                    spec.clip_value,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub(crate) fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        spec: Mxfp4ExpertDecodeGemvSpec,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let per_matrix_bias = arguments.d_transform.per_matrix_bias;
        let gate_act = arguments.d_transform.gate_act;
        let MatmulArguments {
            a,
            b,
            d,
            m,
            n,
            k,
            routing,
            ..
        } = arguments;
        let MatmulA::FullPrecision {
            values: a,
            offset: a_offset,
        } = a
        else {
            return Err(MatmulError::IncompatibleA {
                path: "Mxfp4ExpertDecodeGemv",
                reason: "dedicated MXFP4 decode requires a full-precision activation",
            });
        };
        let MatmulB::Microfloat {
            codes,
            scales,
            outer_scales,
            ..
        } = b
        else {
            return Err(MatmulError::UnsupportedLayout {
                path: "Mxfp4ExpertDecodeGemv",
            });
        };
        let Some(routes) = routing.expert_routes() else {
            return Err(MatmulError::UnsupportedRouting {
                path: "Mxfp4ExpertDecodeGemv",
                reason: "dedicated MXFP4 decode requires direct expert routes",
            });
        };

        let output_width = if spec.fused_gate_up {
            n / 2
        } else {
            n
        };
        let group_count_x = output_width.div_ceil(spec.outputs_per_threadgroup());
        let pipeline = self.get_or_create(encoder.context(), spec)?;
        pipeline.encode(
            codes,
            scales,
            outer_scales,
            (a, a_offset),
            &mut *d,
            per_matrix_bias,
            routes.expert_ids,
            k,
            n,
            m,
            routes.routes_per_token.get(),
            routes.expert_count.get(),
            matches!(routes.input, crate::backends::common::kernel::matmul::ExpertInput::Routes),
            group_count_x,
            gate_act.and_then(|gate_act| gate_act.activation_alpha),
            gate_act.and_then(|gate_act| gate_act.gate_clipping.map(|(min, _)| min)),
            gate_act.and_then(|gate_act| gate_act.gate_clipping.map(|(_, max)| max)),
            gate_act.and_then(|gate_act| gate_act.value_clipping.map(|(min, _)| min)),
            gate_act.and_then(|gate_act| gate_act.value_clipping.map(|(_, max)| max)),
            encoder,
        );
        Ok(())
    }
}
