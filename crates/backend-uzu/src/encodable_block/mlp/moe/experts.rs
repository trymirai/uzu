use parking_lot::Mutex;

use super::router::MoeRoutes;
use crate::{
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{
            MoeFinalizeKernel,
            matmul::{ExpertInput, ExpertRoutes, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
        },
    },
    config::activation::AnyActivation,
    data_type::DataType,
    encodable_block::mlp::gate_act_mul::MlpGateActMulEncodable,
};

pub struct MoeExperts<B: Backend> {
    w13_kernel: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    gate: MlpGateActMulEncodable<B>,
    w2_kernel: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    finalize: <B::Kernels as Kernels>::MoeFinalizeKernel,
    w13: Allocation<B>,
    w2: Allocation<B>,
    up_biases: Allocation<B>,
    down_biases: Allocation<B>,
    model_dim: u32,
    hidden_dim: u32,
    expert_count: u32,
    data_type: DataType,
}

impl<B: Backend> MoeExperts<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &B::Context,
        w13: Allocation<B>,
        w2: Allocation<B>,
        up_biases: Allocation<B>,
        down_biases: Allocation<B>,
        model_dim: u32,
        hidden_dim: u32,
        expert_count: u32,
        activation: AnyActivation,
        gate_clipping: Option<(Option<f32>, Option<f32>)>,
        up_clipping: Option<(Option<f32>, Option<f32>)>,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            w13_kernel: Mutex::new(<B::Kernels as Kernels>::MatmulKernel::new(
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
            w2_kernel: Mutex::new(<B::Kernels as Kernels>::MatmulKernel::new(
                context,
                data_type,
                DataType::F32,
                data_type,
            )?),
            finalize: <B::Kernels as Kernels>::MoeFinalizeKernel::new(context, data_type)?,
            w13,
            w2,
            up_biases,
            down_biases,
            model_dim,
            hidden_dim,
            expert_count,
            data_type,
        })
    }

    pub fn encode(
        &self,
        input: &Allocation<B>,
        routes: &MoeRoutes<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let route_count = routes.route_count();
        let route_metadata = |input, expert_biases| ExpertRoutes {
            expert_ids: &routes.expert_ids,
            routes_per_token: routes.routes_per_token,
            expert_count: std::num::NonZeroU32::new(self.expert_count).expect("MoeBlock validates expert_count"),
            input,
            expert_biases: Some(expert_biases),
        };

        let mut fused_up = encoder.allocate_scratch_for_shape(&[route_count, 2 * self.hidden_dim], DataType::F32)?;
        self.w13_kernel.lock().encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: input,
                    offset: 0,
                },
                b: MatmulB::FullPrecision {
                    b: &self.w13,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut fused_up,
                d_transform: MatmulDOps::none(),
                gather_indices: None,
                expert_routes: Some(route_metadata(ExpertInput::Tokens, &self.up_biases)),
                m: route_count,
                n: 2 * self.hidden_dim,
                k: self.model_dim,
            },
            encoder,
        )?;

        let hidden = self.gate.encode(encoder, &fused_up, route_count)?;
        let mut route_outputs = encoder.allocate_scratch_for_shape(&[route_count, self.model_dim], self.data_type)?;
        self.w2_kernel.lock().encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &hidden,
                    offset: 0,
                },
                b: MatmulB::FullPrecision {
                    b: &self.w2,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut route_outputs,
                d_transform: MatmulDOps::none(),
                gather_indices: None,
                expert_routes: Some(route_metadata(ExpertInput::Routes, &self.down_biases)),
                m: route_count,
                n: self.model_dim,
                k: self.hidden_dim,
            },
            encoder,
        )?;

        let mut output = encoder.allocate_scratch_for_shape(&[routes.token_count, self.model_dim], self.data_type)?;
        self.finalize.encode(
            &routes.weights,
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
