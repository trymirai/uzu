use std::{pin::Pin, sync::Arc};

use tokenizers::Tokenizer;

use crate::{
    traits::backend::{Error, Instance as InstanceTrait},
    types::{
        basic::SamplingParameters,
        session::chat::{ChatConfig, ChatReplyConfig},
    },
};

pub enum TokenStreamOutput {
    LimitReached, // This should be just end of stream
    Token(u64),
}

#[derive(Debug, Clone, Default)]
pub struct TokenStreamMetrics {
    pub num_prefill_forward_passes: usize,
    pub num_decode_forward_passes: usize,
    pub num_tokens_prefilled: usize,
    pub num_tokens_proposed: usize,
    pub num_tokens_accepted: usize,
    pub num_tokens_returned: usize,
}

pub struct StreamConfig {
    pub reply: ChatReplyConfig,
    /// Generated marker after which grammar constraints apply; absent means immediate enforcement.
    pub grammar_trigger: Option<String>,
}

/// Content hash of an image: equal content gives an equal identity.
pub type ImageHash = [u8; 32];

pub struct PreparedImage {
    /// `[t * h * w, channels * temporal_patch_size * patch_size * patch_size]`, flattened
    pub pixel_values: Vec<f32>,
    /// Patch grid `[t, h, w]` before merging
    pub grid_thw: [usize; 3],
    /// Number of image pad tokens the image takes in the prompt
    pub token_count: usize,
}

pub struct StreamImage {
    /// Index in `StreamInput::tokens` of the image's first pad token
    pub start: usize,
    pub image: PreparedImage,
}

#[derive(Default)]
pub struct StreamInput {
    pub tokens: Vec<u64>,
    pub images: Vec<StreamImage>,
}
pub type StreamOutput = TokenStreamOutput;
pub type StreamMetrics = Option<TokenStreamMetrics>;

pub trait Backend: Send + Sync {
    fn instance<'a>(
        &'a self,
        reference: String,
        config: ChatConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn Instance>, Error>> + Send + 'a>>;
}

pub trait Instance:
    InstanceTrait<
        StreamConfig = StreamConfig,
        StreamInput = StreamInput,
        StreamOutput = StreamOutput,
        StreamMetrics = StreamMetrics,
    >
{
    fn with_config(
        &self,
        config: ChatConfig,
    ) -> Box<dyn Instance>;

    fn tokenizer(&self) -> Arc<Tokenizer>;

    fn max_context_length(&self) -> Option<usize>;

    fn stop_token_ids(&self) -> Option<Box<[u64]>>;

    fn sampling_defaults(&self) -> SamplingParameters;

    fn prepare_image(
        &self,
        url: &str,
    ) -> Result<PreparedImage, Error> {
        let _ = url;
        Err("image input is not supported".into())
    }

    fn image_id(
        &self,
        url: &str,
    ) -> Result<ImageHash, Error> {
        let _ = url;
        Err("image input is not supported".into())
    }
}
