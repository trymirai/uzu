use std::iter::repeat_n;

use shoji::types::session::vision::{VisionImage, VisionImageInput};

use crate::{
    backends::Backend,
    engine::language_model::vision::{encode::VisionPrefillError, model::VisionModel},
};

impl<B: Backend> VisionModel<B> {
    pub fn image_token_count(
        &self,
        image: &VisionImage,
    ) -> usize {
        let block_size = self.config.encoder_config.patch_size * self.config.encoder_config.spatial_merge_size;
        ((image.width / block_size) * (image.height / block_size)) as usize
    }

    /// Replaces every image's single pad token with `image_token_count` copies of it, so the input gets a token for each image embedding
    pub fn expand_tokens(
        &self,
        input: &[u64],
        images: &[VisionImageInput],
    ) -> Result<Box<[u64]>, VisionPrefillError<B>> {
        let mut extra_tokens = 0;
        for image in images {
            let image_tokens_count = self.image_token_count(&image.image);
            extra_tokens += image_tokens_count.saturating_sub(1);
        }

        let mut tokens = Vec::with_capacity(input.len() + extra_tokens);
        let mut cursor = 0;
        for image in images {
            if input.len() <= image.start || image.start < cursor {
                return Err(VisionPrefillError::ImagePosition {
                    start: image.start,
                    length: input.len(),
                });
            }

            tokens.extend_from_slice(&input[cursor..image.start]);

            let image_tokens_count = self.image_token_count(&image.image);
            let repeated_tokens = repeat_n(input[image.start], image_tokens_count);
            tokens.extend(repeated_tokens);

            cursor = image.start + 1;
        }
        tokens.extend_from_slice(&input[cursor..]);

        Ok(tokens.into_boxed_slice())
    }
}
