use parking_lot::Mutex;

use super::router::MoeRoutes;
use crate::{
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::{ActivationType, gemm::GemmDTransform},
        kernel::{
            MoeFinalizeKernel,
            matmul::{
                ActivationFormat, ExpertInput, ExpertRoutes, GateActMulDOps, MatmulA, MatmulArguments, MatmulDOps,
                MatmulKernel, MatmulRouting, MatmulShape,
            },
        },
    },
    config::activation::AnyActivation,
    data_type::DataType,
    encodable_block::{linear::LinearInput, mlp::gate_act_mul::MlpGateActMulEncodable, weight_matrix::WeightMatrix},
};

pub struct MoeExperts<B: Backend> {
    up_projection_kernel: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    gate: MlpGateActMulEncodable<B>,
    down_projection_kernel: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    finalize: <B::Kernels as Kernels>::MoeFinalizeKernel,
    up_projection: WeightMatrix<B>,
    down_projection: WeightMatrix<B>,
    up_biases: Allocation<B>,
    down_biases: Allocation<B>,
    model_dim: u32,
    hidden_dim: u32,
    fused_hidden_dim: u32,
    expert_count: u32,
    data_type: DataType,
    silu_alpha: Option<f32>,
    gate_clipping: Option<(f32, f32)>,
    up_clipping: Option<(f32, f32)>,
}

impl<B: Backend> MoeExperts<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &B::Context,
        up_projection: WeightMatrix<B>,
        down_projection: WeightMatrix<B>,
        up_biases: Allocation<B>,
        down_biases: Allocation<B>,
        model_dim: u32,
        hidden_dim: u32,
        fused_hidden_dim: u32,
        expert_count: u32,
        activation: AnyActivation,
        gate_clipping: Option<(Option<f32>, Option<f32>)>,
        up_clipping: Option<(Option<f32>, Option<f32>)>,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        let silu_alpha = (activation.act_type() == ActivationType::SILU).then(|| activation.alpha());
        Ok(Self {
            up_projection_kernel: Mutex::new(<B::Kernels as Kernels>::MatmulKernel::new(
                context,
                data_type,
                data_type,
                DataType::F32,
            )?),
            gate: MlpGateActMulEncodable::new(
                context,
                DataType::F32,
                activation,
                gate_clipping,
                up_clipping,
                hidden_dim,
                None,
            )?,
            down_projection_kernel: Mutex::new(<B::Kernels as Kernels>::MatmulKernel::new(
                context,
                data_type,
                DataType::F32,
                data_type,
            )?),
            finalize: <B::Kernels as Kernels>::MoeFinalizeKernel::new(context, data_type)?,
            up_projection,
            down_projection,
            up_biases,
            down_biases,
            model_dim,
            hidden_dim,
            fused_hidden_dim,
            expert_count,
            data_type,
            silu_alpha,
            gate_clipping: clipping_bounds(gate_clipping),
            up_clipping: clipping_bounds(up_clipping),
        })
    }

    fn encode_hidden(
        &self,
        input: &Allocation<B>,
        routes: &MoeRoutes<B>,
        route_count: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut fused_up = encoder.allocate_scratch_for_shape(&[route_count, self.fused_hidden_dim], DataType::F32)?;
        self.up_projection_kernel.lock().encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: input,
                    offset: 0,
                },
                b: self.up_projection.matmul_b(),
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut fused_up,
                d_transform: MatmulDOps {
                    per_matrix_bias: Some(&self.up_biases),
                    ..MatmulDOps::none()
                },
                routing: MatmulRouting::Experts(ExpertRoutes {
                    expert_ids: &routes.expert_ids,
                    routes_per_token: routes.routes_per_token,
                    expert_count: std::num::NonZeroU32::new(self.expert_count)
                        .expect("MoeBlock validates expert_count"),
                    input: ExpertInput::Tokens,
                }),
                m: route_count,
                n: self.fused_hidden_dim,
                k: self.model_dim,
            },
            encoder,
        )?;
        let gate_input = self.gate.encode_for_linear(encoder, &fused_up, route_count, ActivationFormat::Bf16)?;
        let hidden = match gate_input {
            LinearInput::FullPrecision(hidden) => hidden,
            _ => unreachable!("BF16 activation format always yields full-precision hidden states"),
        };
        Ok(hidden)
    }

    pub fn encode(
        &self,
        input: &Allocation<B>,
        routes: &MoeRoutes<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let route_count = routes.route_count();
        let route_metadata = |input| ExpertRoutes {
            expert_ids: &routes.expert_ids,
            routes_per_token: routes.routes_per_token,
            expert_count: std::num::NonZeroU32::new(self.expert_count).expect("MoeBlock validates expert_count"),
            input,
        };

        let mut hidden = None;
        if let Some(alpha) = self.silu_alpha {
            let b = self.up_projection.matmul_b();
            let fused_shape = MatmulShape {
                m: route_count,
                n: self.fused_hidden_dim,
                k: self.model_dim,
                b_transpose: true,
                b_leading_dimension: None,
                b_prologue: b.b_prologue(),
                b_bits: b.bits_per_b(),
                b_group_size: b.group_size(),
                b_microfloat: b.microfloat_metadata(),
                signed_codes: b.signed_codes(),
                a_full_precision: true,
                sparse_readout: false,
                expert_routed: true,
                expert_bias: true,
                d_transform: GemmDTransform::empty(),
            };
            let mut up_projection_kernel = self.up_projection_kernel.lock();
            if up_projection_kernel.supports_fused_gate_act(&fused_shape) {
                let mut fused_hidden =
                    encoder.allocate_scratch_for_shape(&[route_count, self.hidden_dim], DataType::F32)?;
                up_projection_kernel.encode(
                    MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values: input,
                            offset: 0,
                        },
                        b,
                        b_leading_dimension: None,
                        b_transpose: true,
                        d: &mut fused_hidden,
                        d_transform: MatmulDOps {
                            per_matrix_bias: Some(&self.up_biases),
                            gate_act: Some(GateActMulDOps {
                                activation_alpha: (alpha != 1.0).then_some(alpha),
                                gate_clipping: self.gate_clipping,
                                value_clipping: self.up_clipping,
                            }),
                            ..MatmulDOps::none()
                        },
                        routing: MatmulRouting::Experts(route_metadata(ExpertInput::Tokens)),
                        m: route_count,
                        n: self.fused_hidden_dim,
                        k: self.model_dim,
                    },
                    encoder,
                )?;
                hidden = Some(fused_hidden);
            }
        }
        let hidden = match hidden {
            Some(hidden) => hidden,
            None => self.encode_hidden(input, routes, route_count, encoder)?,
        };
        let mut route_outputs = encoder.allocate_scratch_for_shape(&[route_count, self.model_dim], self.data_type)?;
        self.down_projection_kernel.lock().encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &hidden,
                    offset: 0,
                },
                b: self.down_projection.matmul_b(),
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut route_outputs,
                d_transform: MatmulDOps {
                    per_matrix_bias: Some(&self.down_biases),
                    ..MatmulDOps::none()
                },
                routing: MatmulRouting::Experts(route_metadata(ExpertInput::Routes)),
                m: route_count,
                n: self.model_dim,
                k: self.hidden_dim,
            },
            encoder,
        )?;

        let mut output = encoder.allocate_scratch_for_shape(&[routes.token_count, self.model_dim], self.data_type)?;
        self.finalize.encode(
            &routes.route_weights,
            &route_outputs,
            &mut output,
            routes.token_count,
            self.model_dim,
            routes.routes_per_token.get(),
            encoder,
        );
        Ok(output)
    }
}

fn clipping_bounds(clipping: Option<(Option<f32>, Option<f32>)>) -> Option<(f32, f32)> {
    let (min, max) = clipping?;
    if min.is_none() && max.is_none() {
        return None;
    }
    Some((min.unwrap_or(f32::NEG_INFINITY), max.unwrap_or(f32::INFINITY)))
}
