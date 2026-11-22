use std::{mem::size_of, path::Path, sync::Arc};

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, Context, Kernels, gpu_types::trie::TrieNode as GpuTrieNode,
        kernel::LinearizeChainKernel,
    },
    config::speculator::dspark::DSparkSpeculatorConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        decoder::Decoder,
        dspark::{DSpark, DSparkState},
        sampling::{PRng, Sampling, SamplingMethod},
        transformer::HiddenFeature,
    },
    safetensors::ParameterLoader,
    speculators::dflash_tfm::{DFlashSpeculatorLoadError, DFlashTfmTreeShape, DFlashTreeError, TreeDraft, TreeProposal},
};

/// Proposes chains with a DSpark block drafter and its Markov head.
pub struct DSparkSpeculator<B: Backend> {
    context: Arc<B::Context>,
    dspark: DSpark<B>,
    sampling: Sampling<B>,
    linearize_chain: [<B::Kernels as Kernels>::LinearizeChainKernel; 2],
    hidden_features: Box<[HiddenFeature]>,
    shape: DFlashTfmTreeShape,
}

impl<B: Backend> DSparkSpeculator<B> {
    pub async fn new(
        model_path: &Path,
        config: DSparkSpeculatorConfig,
        shape: DFlashTfmTreeShape,
        target: &Decoder<B>,
        context: Arc<B::Context>,
    ) -> Result<Self, DFlashSpeculatorLoadError<B>> {
        let visits = target.layer_visit_count();
        let hidden_features = config
            .target_feature_indices
            .iter()
            .map(|&index| match index {
                index if index == visits => Ok(HiddenFeature::Final),
                index if index < visits => Ok(HiddenFeature::BeforeVisit(index)),
                index => Err(DFlashSpeculatorLoadError::TargetMismatch(format!(
                    "tap {index} past the target's {visits} layer visits"
                ))),
            })
            .collect::<Result<Box<[_]>, _>>()?;

        let weight_loader = ParameterLoader::new(model_path.join("model.safetensors"), &*context).await?;
        let dspark = DSpark::new(&config, target, &weight_loader.tree(), &*context).await?;
        weight_loader.assert_all_tensors_read()?;

        let linearize_chain = [
            <B::Kernels as Kernels>::LinearizeChainKernel::new(&*context, false)
                .map_err(DFlashSpeculatorLoadError::Backend)?,
            <B::Kernels as Kernels>::LinearizeChainKernel::new(&*context, true)
                .map_err(DFlashSpeculatorLoadError::Backend)?,
        ];
        Ok(Self {
            sampling: Sampling::new(target.embedding().vocab_size(), DataType::BF16),
            context,
            dspark,
            linearize_chain,
            hidden_features,
            shape,
        })
    }

    pub fn hidden_features(&self) -> &[HiddenFeature] {
        &self.hidden_features
    }

    pub fn empty_state(
        &self,
        context_capacity: u32,
    ) -> Result<DSparkState<B>, B::Error> {
        self.dspark.empty_state(context_capacity, &self.context)
    }

    pub fn encode_accept(
        &self,
        state: &mut DSparkState<B>,
        target_features: impl ExactSizeIterator<Item = impl BufferRef<Backend = B>>,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), DFlashTreeError<B>> {
        Ok(self.dspark.encode_accept(target_features, accepted_indices, state, command_buffer)?)
    }

    /// The shape's chain, cut to the block's proposal positions and to `max_depth` tokens with the root.
    pub fn make_shape(
        &self,
        max_depth: Option<u32>,
    ) -> Option<DFlashTfmTreeShape> {
        if max_depth.is_some_and(|max_depth| max_depth < 2) {
            return None;
        }
        let mut shape = self.shape.clone();
        let depth = shape.tree_budget.min(self.dspark.block_size()).min(max_depth.unwrap_or(u32::MAX));
        shape.tree_budget = depth;
        shape.max_tree_depth = depth;
        Some(shape)
    }

    pub fn encode_draft(
        &self,
        state: &mut DSparkState<B>,
        target_output_token: impl BufferRef<Backend = B>,
        target: &Decoder<B>,
        shape: &DFlashTfmTreeShape,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TreeDraft<B>, DFlashTreeError<B>> {
        let depth = shape.tree_budget;
        let root_position = state.context_length();
        let dflash_output = self.dspark.encode_draft(target, target_output_token, depth, state, command_buffer)?;
        Ok(TreeDraft {
            dflash_output,
            dflash_depth: depth,
            root_position,
        })
    }

    pub fn propose_tree(
        &self,
        draft: TreeDraft<B>,
        state: &DSparkState<B>,
        target_output_token: impl BufferRef<Backend = B, Buffer = B::GlobalGpuBuffer>,
        sampling_method: &SamplingMethod,
        prng: &PRng,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TreeProposal<B>, DFlashTreeError<B>> {
        let chain_length = draft.dflash_depth - 1;
        let root_position = draft.root_position;
        let with_seeds = matches!(sampling_method, SamplingMethod::Stochastic { .. });
        // The token after the node at depth d is sampled with that node's seed, as the target samples it.
        let seeds = with_seeds.then(|| {
            (0..chain_length).map(|depth| prng.derive(root_position as u64 + depth as u64)).collect::<Box<[u64]>>()
        });
        let sampled = self.dspark.encode_markov_chain(
            &draft.dflash_output,
            target_output_token,
            chain_length,
            state,
            &self.sampling,
            sampling_method,
            seeds.as_deref(),
            command_buffer,
        )?;

        let budget = chain_length + 1;
        let context = &*self.context;
        let mut nodes =
            context.create_buffer_gpu(budget as usize * size_of::<GpuTrieNode>()).map_err(DFlashTreeError::Backend)?;
        let mut token_ids =
            context.create_buffer_gpu(budget as usize * size_of::<u32>()).map_err(DFlashTreeError::Backend)?;
        let mut trie_seeds = if with_seeds {
            Some(context.create_buffer_gpu(budget as usize * size_of::<u64>()).map_err(DFlashTreeError::Backend)?)
        } else {
            None
        };
        self.linearize_chain[usize::from(with_seeds)].encode(
            &sampled,
            target_output_token,
            &mut nodes,
            &mut token_ids,
            trie_seeds.as_mut(),
            chain_length,
            budget,
            prng.seed(),
            root_position,
            command_buffer,
        );
        Ok(TreeProposal {
            topology: BatchTopology::new(Some(nodes), budget, false, true),
            token_ids,
            seeds: trie_seeds,
        })
    }
}
