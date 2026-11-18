use std::iter;

use shoji::traits::backend::{Error as ShojiBackendError, chat_token::StreamInput as ChatTokenStreamInput};
use uzu_engine::{
    backends::Backend,
    engine::language_model::{LanguageModel, stream::VisionModelInputImage},
};

use crate::vision::{VisionPreprocessingConfig, preprocess};

pub struct PreparedImage {
    /// `[t * h * w, channels * temporal_patch_size * patch_size * patch_size]`, flattened
    pixel_values: Vec<f32>,
    /// Patch grid `[t, h, w]` before merging
    grid_thw: [usize; 3],
    /// Number of image pad tokens the image takes in the prompt
    token_count: usize,
}

pub fn prepare_image<B: Backend>(
    model: &LanguageModel<B>,
    url: &str,
) -> Result<PreparedImage, ShojiBackendError> {
    let vision_config = model.vision_config().ok_or("model doesn't support image input")?;
    let encoder_config = &vision_config.encoder_config;
    let config = VisionPreprocessingConfig {
        patch_size: encoder_config.patch_size as usize,
        merge_size: encoder_config.spatial_merge_size as usize,
        temporal_patch_size: encoder_config.temporal_patch_size as usize,
        shortest_edge: vision_config.preprocessing_config.min_pixels as usize,
        longest_edge: vision_config.preprocessing_config.max_pixels as usize,
        image_mean: vision_config.preprocessing_config.image_mean,
        image_std: vision_config.preprocessing_config.image_std,
    };

    let bytes = std::fs::read(url)?;
    let output = preprocess(&bytes, &config)?;
    let token_count = vision_config.image_token_count(output.image_grid_thw.map(|size| size as u32)) as usize;
    Ok(PreparedImage {
        pixel_values: output.pixel_values,
        grid_thw: output.image_grid_thw,
        token_count,
    })
}

/// Repeats each image's pad token `token_count` times and points the model images at the expanded pads.
pub fn expand_images<'a>(
    input: &ChatTokenStreamInput,
    prepared_images: &'a [PreparedImage],
) -> (Vec<u64>, Box<[VisionModelInputImage<'a>]>) {
    let mut tokens =
        Vec::with_capacity(input.tokens.len() + prepared_images.iter().map(|image| image.token_count).sum::<usize>());
    let mut images = Vec::with_capacity(prepared_images.len());
    let mut copied = 0;
    for (image, prepared_image) in input.images.iter().zip(prepared_images) {
        tokens.extend_from_slice(&input.tokens[copied..image.start]);
        images.push(VisionModelInputImage {
            start: tokens.len() as u32,
            pixel_values: &prepared_image.pixel_values,
            grid_thw: prepared_image.grid_thw.map(|size| size as u32),
        });
        tokens.extend(iter::repeat_n(input.tokens[image.start], prepared_image.token_count));
        copied = image.start + 1;
    }
    tokens.extend_from_slice(&input.tokens[copied..]);
    (tokens, images.into_boxed_slice())
}
