use std::num::NonZeroU32;

use crate::{
    backends::common::{Allocation, Backend, Encoder, Kernels, kernel::MoeRouterTopKKernel},
    data_type::DataType,
};

pub struct MoeRoutes<B: Backend> {
    pub expert_ids: Allocation<B>,
    pub weights: Allocation<B>,
    pub token_count: u32,
    pub routes_per_token: NonZeroU32,
}

impl<B: Backend> MoeRoutes<B> {
    pub fn route_count(&self) -> u32 {
        self.token_count * self.routes_per_token.get()
    }
}

pub struct MoeRouter<B: Backend> {
    kernel: <B::Kernels as Kernels>::MoeRouterTopKKernel,
    weights: Allocation<B>,
    biases: Allocation<B>,
    model_dim: u32,
    expert_count: u32,
    routes_per_token: NonZeroU32,
    renormalize: bool,
    data_type: DataType,
}

impl<B: Backend> MoeRouter<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &B::Context,
        weights: Allocation<B>,
        biases: Allocation<B>,
        model_dim: u32,
        expert_count: u32,
        routes_per_token: u32,
        renormalize: bool,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            kernel: <B::Kernels as Kernels>::MoeRouterTopKKernel::new(
                context, data_type, true, false, false, false, false,
            )?,
            weights,
            biases,
            model_dim,
            expert_count,
            routes_per_token: NonZeroU32::new(routes_per_token)
                .expect("MoeBlock validates that routes_per_token is nonzero"),
            renormalize,
            data_type,
        })
    }

    pub fn route(
        &self,
        input: &Allocation<B>,
        token_count: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<MoeRoutes<B>, B::Error> {
        let routes_per_token = self.routes_per_token.get();
        let mut expert_ids = encoder.allocate_scratch_for_shape(&[token_count, routes_per_token], DataType::I32)?;
        let mut weights = encoder.allocate_scratch_for_shape(&[token_count, routes_per_token], self.data_type)?;
        encoder.encode_fill(&mut expert_ids, 0xFF);
        self.kernel.encode(
            input,
            &self.weights,
            Some(&self.biases),
            None::<&Allocation<B>>,
            None::<&Allocation<B>>,
            &mut expert_ids,
            &mut weights,
            token_count,
            self.model_dim,
            self.expert_count,
            routes_per_token,
            self.renormalize,
            None::<f32>,
            None::<f32>,
            encoder,
        );
        Ok(MoeRoutes {
            expert_ids,
            weights,
            token_count,
            routes_per_token: self.routes_per_token,
        })
    }
}
