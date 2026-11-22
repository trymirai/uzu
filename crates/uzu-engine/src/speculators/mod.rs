use std::{collections::HashMap, path::Path, sync::Arc};

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, Context},
    config::speculator::AnySpeculatorConfig,
    encodable_block::{
        decoder::Decoder,
        dflash::DFlashState,
        dspark::DSparkState,
        sampling::{PRng, SamplingMethod},
        transformer::HiddenFeature,
    },
    utils::file,
};

pub mod dflash_tfm;
pub mod dspark;

use dflash_tfm::{
    DFlashSpeculatorLoadError, DFlashTfmSpeculator, DFlashTfmTreeShape, DFlashTreeError, TreeDraft, TreeProposal,
};
use dspark::DSparkSpeculator;

pub enum AnySpeculator<B: Backend> {
    DFlash(DFlashTfmSpeculator<B>),
    DSpark(DSparkSpeculator<B>),
}

pub enum AnySpeculatorState<B: Backend> {
    DFlash(DFlashState<B>),
    DSpark(DSparkState<B>),
}

const MISMATCHED_STATE: &str = "speculator state of another speculator";

impl<B: Backend> AnySpeculator<B> {
    /// Loads the speculator in `model_path`, or none when `shapes.json` has no shape for this device.
    pub async fn load(
        model_path: &Path,
        target: &Decoder<B>,
        context: Arc<B::Context>,
    ) -> Result<Option<Self>, DFlashSpeculatorLoadError<B>> {
        let mut shapes: HashMap<String, DFlashTfmTreeShape> =
            serde_json::from_slice(&file::read(model_path.join("shapes.json")).await?)?;
        let Some(shape) = context
            .device_name()
            .and_then(|device_name| shapes.remove(device_name))
            .or_else(|| shapes.remove("default"))
        else {
            return Ok(None);
        };
        let config: AnySpeculatorConfig = serde_json::from_slice(&file::read(model_path.join("config.json")).await?)?;
        Ok(Some(match config {
            AnySpeculatorConfig::DFlashSpeculatorConfig(config) => Self::DFlash(
                DFlashTfmSpeculator::new(model_path, config, shape, target.embedding(), context).await?,
            ),
            AnySpeculatorConfig::DSparkSpeculatorConfig(config) => {
                Self::DSpark(DSparkSpeculator::new(model_path, config, shape, target, context).await?)
            },
        }))
    }

    pub fn hidden_features(&self) -> &[HiddenFeature] {
        match self {
            Self::DFlash(speculator) => speculator.hidden_features(),
            Self::DSpark(speculator) => speculator.hidden_features(),
        }
    }

    pub fn empty_state(
        &self,
        context_capacity: u32,
    ) -> Result<AnySpeculatorState<B>, B::Error> {
        Ok(match self {
            Self::DFlash(speculator) => AnySpeculatorState::DFlash(speculator.empty_state(context_capacity)?),
            Self::DSpark(speculator) => AnySpeculatorState::DSpark(speculator.empty_state(context_capacity)?),
        })
    }

    pub fn encode_accept(
        &self,
        state: &mut AnySpeculatorState<B>,
        target_features: impl ExactSizeIterator<Item = impl BufferRef<Backend = B>>,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), DFlashTreeError<B>> {
        match (self, state) {
            (Self::DFlash(speculator), AnySpeculatorState::DFlash(state)) => {
                speculator.encode_accept(state, target_features, accepted_indices, command_buffer)
            },
            (Self::DSpark(speculator), AnySpeculatorState::DSpark(state)) => {
                speculator.encode_accept(state, target_features, accepted_indices, command_buffer)
            },
            _ => panic!("{MISMATCHED_STATE}"),
        }
    }

    pub fn make_shape(
        &self,
        max_depth: Option<u32>,
    ) -> Option<DFlashTfmTreeShape> {
        match self {
            Self::DFlash(speculator) => speculator.make_shape(max_depth),
            Self::DSpark(speculator) => speculator.make_shape(max_depth),
        }
    }

    pub fn encode_draft(
        &self,
        state: &mut AnySpeculatorState<B>,
        target_output_token: impl BufferRef<Backend = B>,
        target: &Decoder<B>,
        shape: &DFlashTfmTreeShape,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TreeDraft<B>, DFlashTreeError<B>> {
        match (self, state) {
            (Self::DFlash(speculator), AnySpeculatorState::DFlash(state)) => {
                speculator.encode_draft(state, target_output_token, target.embedding(), shape, command_buffer)
            },
            (Self::DSpark(speculator), AnySpeculatorState::DSpark(state)) => {
                speculator.encode_draft(state, target_output_token, target, shape, command_buffer)
            },
            _ => panic!("{MISMATCHED_STATE}"),
        }
    }

    pub fn propose_tree(
        &self,
        draft: TreeDraft<B>,
        state: &AnySpeculatorState<B>,
        target_output_norm: impl BufferRef<Backend = B>,
        target_output_token: impl BufferRef<Backend = B, Buffer = B::GlobalGpuBuffer>,
        target: &Decoder<B>,
        shape: DFlashTfmTreeShape,
        sampling_method: &SamplingMethod,
        prng: &PRng,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TreeProposal<B>, DFlashTreeError<B>> {
        match (self, state) {
            (Self::DFlash(speculator), AnySpeculatorState::DFlash(_)) => speculator.propose_tree(
                draft,
                target_output_norm,
                target_output_token,
                target.embedding(),
                shape,
                sampling_method,
                prng,
                command_buffer,
            ),
            (Self::DSpark(speculator), AnySpeculatorState::DSpark(state)) => {
                speculator.propose_tree(draft, state, target_output_token, sampling_method, prng, command_buffer)
            },
            _ => panic!("{MISMATCHED_STATE}"),
        }
    }
}
