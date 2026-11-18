use std::range::Range;

use derive_more::Debug;
use itertools::iproduct;
use thiserror::Error;

use super::{VisionModel, VisionModelEncodeError};
use crate::{
    backends::common::{Backend, CommandBuffer},
    encodable_block::{embedding::EmbeddingReplacement, mixer::attention::rope::MultimodalRoPEPositions},
};

/// An image in the prefill input: its pad tokens are already in the input from `start`, and the vision encoder output replaces their embeddings.
pub struct VisionModelInputImage<'a> {
    /// Index in the input of the image's first pad token
    pub start: u32,
    /// Preprocessed patches of a single frame: `[h * w, channels * temporal_patch_size * patch_size * patch_size]`
    pub pixel_values: &'a [f32],
    /// Patch grid `[t, h, w]` before merging
    pub grid_thw: [u32; 3],
}

pub struct VisionModelEncodedImage<B: Backend> {
    /// Indices of the image's pad tokens in the prefill input
    tokens: Range<usize>,
    /// Merged patch grid: the pad tokens are its `height x width` cells, row by row
    height: u32,
    width: u32,
    /// Vision encoder output: one row per pad token
    embeddings: B::ScratchBuffer,
}

/// The images of a prefill input, encoded by the vision encoder.
pub struct VisionModelPrefilledImages<B: Backend> {
    images: Box<[VisionModelEncodedImage<B>]>,
    /// RoPE positions `[t, h, w]` of every input token
    rope_positions: Box<[[u32; 3]]>,
    /// RoPE position of the token after the input
    next_rope_position: u32,
}

impl<B: Backend> VisionModelPrefilledImages<B> {
    /// Multimodal RoPE positions of the prefill chunk at `chunk` of the input.
    pub fn multimodal_rope(
        &self,
        chunk: Range<usize>,
    ) -> MultimodalRoPEPositions<'_> {
        MultimodalRoPEPositions {
            sections: [11, 11, 10],
            positions: &self.rope_positions[chunk.start..chunk.end],
        }
    }

    /// RoPE position of the token after the prefill input.
    pub fn next_rope_position(&self) -> u32 {
        self.next_rope_position
    }

    /// Replacements of the image pad token embeddings in the prefill chunk at `chunk` of the input.
    pub fn embedding_replacements(
        &self,
        chunk: Range<usize>,
    ) -> Box<[EmbeddingReplacement<'_, B>]> {
        self.images
            .iter()
            .filter_map(|image| {
                let start = image.tokens.start.max(chunk.start);
                let end = image.tokens.end.min(chunk.end);
                (start < end).then(|| EmbeddingReplacement {
                    source: &image.embeddings,
                    source_row: (start - image.tokens.start) as u32,
                    row: (start - chunk.start) as u32,
                    rows: (end - start) as u32,
                })
            })
            .collect()
    }
}

#[derive(Debug, Error)]
pub enum VisionModelPrefillError<B: Backend> {
    #[error("Vision encoder error: {0}")]
    Encode(#[from] VisionModelEncodeError<B>),
    #[error("Image tokens {tokens:?} overlap the previous image or exceed the input of {input_length} tokens")]
    InvalidTokens {
        tokens: Range<usize>,
        input_length: usize,
    },
    #[error("Model has no vision encoder")]
    NoVisionEncoder,
    #[error("Unsupported patch grid {0:?}")]
    UnsupportedGrid([u32; 3]),
}

impl<B: Backend> VisionModel<B> {
    pub fn prefill(
        &self,
        images: &[VisionModelInputImage],
        input_length: usize,
        first_rope_position: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<VisionModelPrefilledImages<B>, VisionModelPrefillError<B>> {
        let merge_size = self.config().encoder_config.spatial_merge_size;
        let mut encoded_images: Vec<VisionModelEncodedImage<B>> = Vec::with_capacity(images.len());

        for image in images {
            let [frames, height, width] = image.grid_thw;
            if frames != 1 || !height.is_multiple_of(merge_size) || !width.is_multiple_of(merge_size) {
                return Err(VisionModelPrefillError::UnsupportedGrid(image.grid_thw));
            }

            let start = image.start as usize;
            let tokens = Range::from(start..start + self.config().image_token_count(image.grid_thw) as usize);
            let previous_end = encoded_images.last().map_or(0, |previous| previous.tokens.end);
            if tokens.start < previous_end || tokens.end > input_length {
                return Err(VisionModelPrefillError::InvalidTokens {
                    tokens,
                    input_length,
                });
            }

            let embeddings = self.encode(image.pixel_values, height, width, command_buffer)?;
            let image = VisionModelEncodedImage {
                tokens,
                height: height / merge_size,
                width: width / merge_size,
                embeddings,
            };
            encoded_images.push(image);
        }

        let (rope_positions, next_rope_position) =
            Self::rope_positions(&encoded_images, input_length, first_rope_position);
        let images = VisionModelPrefilledImages {
            images: encoded_images.into(),
            rope_positions,
            next_rope_position,
        };

        Ok(images)
    }

    /// RoPE positions `[t, h, w]` of an `input_length`-token input from `first_position`, as HF `get_rope_index` does:
    /// a text token takes the next position on all axes, and an image starting at position `p` gets
    /// `[p, p + row, p + column]`, with the text after it continuing from `p + max(height, width)`.
    /// Also returns the position after the input.
    fn rope_positions(
        images: &[VisionModelEncodedImage<B>],
        input_length: usize,
        first_position: u32,
    ) -> (Box<[[u32; 3]]>, u32) {
        let mut positions = Vec::with_capacity(input_length);
        let mut position = first_position;
        for image in images.iter().map(Some).chain([None]) {
            let text_end = image.map_or(input_length, |image| image.tokens.start);
            while positions.len() < text_end {
                positions.push([position; 3]);
                position += 1;
            }

            if let Some(image) = image {
                positions.extend(
                    iproduct!(0..image.height, 0..image.width)
                        .map(|(row, column)| [position, position + row, position + column]),
                );
                position += image.height.max(image.width);
            }
        }
        (positions.into(), position)
    }
}
