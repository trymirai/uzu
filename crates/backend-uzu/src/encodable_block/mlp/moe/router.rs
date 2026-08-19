use std::num::NonZeroU32;

use crate::{
    backends::common::{Allocation, Backend, Encoder, Kernels, kernel::MoeRouterTopKKernel},
    data_type::DataType,
};

fn checked_route_count(
    token_count: u32,
    routes_per_token: u32,
) -> Option<u32> {
    token_count.checked_mul(routes_per_token)
}

pub struct MoeRoutes<B: Backend> {
    pub expert_ids: Allocation<B>,
    pub route_weights: Allocation<B>,
    pub token_count: u32,
    pub routes_per_token: NonZeroU32,
    route_count: u32,
}

impl<B: Backend> MoeRoutes<B> {
    pub(super) fn from_parts(
        expert_ids: Allocation<B>,
        route_weights: Allocation<B>,
        token_count: u32,
        routes_per_token: NonZeroU32,
    ) -> Option<Self> {
        let route_count = checked_route_count(token_count, routes_per_token.get())?;
        Some(Self {
            expert_ids,
            route_weights,
            token_count,
            routes_per_token,
            route_count,
        })
    }

    pub fn route_count(&self) -> u32 {
        self.route_count
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
        // `Mlp::encode` can only return `B::Error`, so it cannot surface a
        // configuration error for a caller-supplied token count. Check before
        // allocation or dispatch; exceeding u32 route indexing is a caller
        // contract violation.
        checked_route_count(token_count, routes_per_token).expect("MoE route count must fit in u32");
        let mut expert_ids = encoder.allocate_scratch_for_shape(&[token_count, routes_per_token], DataType::I32)?;
        let mut route_weights = encoder.allocate_scratch_for_shape(&[token_count, routes_per_token], self.data_type)?;
        // No prefill of `expert_ids`: the TopK kernels write every slot
        // unconditionally (invalid winners become id -1 with weight 0).
        self.kernel.encode(
            input,
            &self.weights,
            Some(&self.biases),
            None::<&Allocation<B>>,
            None::<&Allocation<B>>,
            &mut expert_ids,
            &mut route_weights,
            token_count,
            self.model_dim,
            self.expert_count,
            routes_per_token,
            self.renormalize,
            None::<f32>,
            None::<f32>,
            encoder,
        );
        Ok(MoeRoutes::from_parts(expert_ids, route_weights, token_count, self.routes_per_token)
            .expect("route count was checked before allocating router outputs"))
    }
}

#[cfg(test)]
mod tests {
    use proc_macros::uzu_test;

    use super::checked_route_count;

    #[uzu_test]
    fn route_count_multiplication_is_checked() {
        assert_eq!(checked_route_count(33, 2), Some(66));
        assert_eq!(checked_route_count(u32::MAX, 2), None);
    }
}
