use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder, Kernels,
        kernel::{
            FullPrecisionEmbeddingLookupKernel, ManualKernels, PLECombineKernel,
            matmul::{MatmulArgumentC, MatmulArguments, MatmulKernel},
        },
    },
    config::PLEModelConfig,
    encodable_block::model_extension::{ModelExtensionError, validate_tensor},
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

pub struct PerLayerEmbedding<B: Backend> {
    token_embedding: Rc<RefCell<B::Buffer>>,
    model_projection: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
    model_projection_weights: Rc<RefCell<B::Buffer>>,
    projection_norm_scales: Rc<RefCell<B::Buffer>>,
    lookup_kernel: <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel,
    combine_kernel: <B::Kernels as Kernels>::PLECombineKernel,
    config: PLEModelConfig,
    model_dim: usize,
}

impl<B: Backend> PerLayerEmbedding<B> {
    pub fn new(
        context: &B::Context,
        model_dim: usize,
        config: PLEModelConfig,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, ModelExtensionError<B>> {
        let activation_type: DataType = config.linear_config.activation_precision().into();
        let total_ple_dim = config.num_layers * config.ple_dim;

        let token_embedding = parameter_tree.leaf_array("token_embedding")?;
        validate_tensor::<B>(
            token_embedding.shape(),
            token_embedding.data_type(),
            &[config.ple_vocab_size, total_ple_dim],
            activation_type,
        )?;

        let projection_tree = parameter_tree.subtree("model_projection")?;
        let projection_weights = projection_tree.leaf_array("weights")?;
        validate_tensor::<B>(
            projection_weights.shape(),
            projection_weights.data_type(),
            &[total_ple_dim, model_dim],
            activation_type,
        )?;

        let projection_norm_tree = parameter_tree.subtree("projection_norm")?;
        let projection_norm_scales = projection_norm_tree.leaf_array("scales")?;
        validate_tensor::<B>(
            projection_norm_scales.shape(),
            projection_norm_scales.data_type(),
            &[config.ple_dim],
            projection_norm_scales.data_type(),
        )?;

        let lookup_kernel = <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, activation_type)
            .map_err(ModelExtensionError::BackendError)?;
        let model_projection = <B::Kernels as ManualKernels>::MatmulKernel::new(context, activation_type)?;
        let combine_kernel = <B::Kernels as Kernels>::PLECombineKernel::new(
            context,
            activation_type,
            projection_norm_scales.data_type(),
        )
        .map_err(ModelExtensionError::BackendError)?;

        Ok(Self {
            token_embedding: token_embedding.buffer(),
            model_projection: RefCell::new(model_projection),
            model_projection_weights: projection_weights.buffer(),
            projection_norm_scales: projection_norm_scales.buffer(),
            lookup_kernel,
            combine_kernel,
            config,
            model_dim,
        })
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) {
        let batch_dim = state.active_row_count();
        if batch_dim == 0 {
            return;
        }

        let token_ids = state.array(ArrayId::TokenIds);
        let token_ids_buffer = token_ids.buffer();
        let ple_token = state.array(ArrayId::PleToken);
        let ple_token_buffer = ple_token.buffer();
        let total_ple_dim = self.config.num_layers * self.config.ple_dim;

        self.lookup_kernel.encode(
            token_ids_buffer.borrow().deref(),
            self.token_embedding.borrow().deref(),
            ple_token_buffer.borrow_mut().deref_mut(),
            batch_dim as u32,
            self.config.ple_vocab_size as u32,
            total_ple_dim as u32,
            self.config.ple_embed_scale,
            encoder,
        );

        let main = state.array(ArrayId::Main);
        let ple_model = state.array(ArrayId::PleModel);
        self.model_projection.borrow_mut().encode(
            state.context(),
            MatmulArguments {
                a: main.buffer().borrow().deref(),
                a_offset: 0,
                b: self.model_projection_weights.borrow().deref(),
                ab_scale: 1.0,
                c: MatmulArgumentC::None,
                d: ple_model.buffer().borrow_mut().deref_mut(),
                batch_dim: batch_dim as u32,
                input_dim: self.model_dim as u32,
                output_dim: total_ple_dim as u32,
            },
            encoder,
        );

        let ple_combined = state.array(ArrayId::PleCombined);
        self.combine_kernel.encode(
            ple_token.buffer().borrow().deref(),
            ple_model.buffer().borrow().deref(),
            self.projection_norm_scales.borrow().deref(),
            ple_combined.buffer().borrow_mut().deref_mut(),
            batch_dim as u32,
            self.config.num_layers as u32,
            self.config.ple_dim as u32,
            self.config.model_projection_scale,
            self.config.input_scale,
            self.config.norm_config.epsilon,
            self.config.norm_config.scale_offset.unwrap_or(0.0),
            encoder,
        );
    }
}
