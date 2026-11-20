use std::range::Range;

use shoji::types::session::vision::{VisionImage, VisionImageInput};
use thiserror::Error;

use crate::{
    backends::{
        Backend,
        common::{CommandBuffer, CommandBufferEncoding, kernel::VisionPatchifyKernel},
    },
    config::vision::VisionRoPEConfig,
    encodable_block::{
        decoder::EmbeddingOverride,
        linear::LinearEncodeError,
        mixer::attention::{mrope::MRoPEPositions, rope::PrecalculatedRoPE},
        mlp::MlpEncodeError,
        vision_layer::VisionLayerEncodeError,
    },
    engine::language_model::vision::{
        model::VisionModel,
        patch_grid::{VisionImageError, vision_patch_coordinates},
    },
};

pub struct VisionPrefilledImage<B: Backend> {
    embeddings: B::ScratchBuffer,
    start: usize,
    width: u32,
    height: u32,
}

pub struct VisionPrefillResult<B: Backend> {
    images: Box<[VisionPrefilledImage<B>]>,
}

impl<B: Backend> VisionPrefillResult<B> {
    pub fn embedding_overrides(
        &self,
        batch: Range<usize>,
    ) -> Vec<EmbeddingOverride<'_, B>> {
        let mut overrides = Vec::new();
        for image in self.images.iter() {
            let image_end = image.start + (image.width * image.height) as usize;
            let start = image.start.max(batch.start);
            let end = image_end.min(batch.end);
            if start < end {
                overrides.push(EmbeddingOverride {
                    source: &image.embeddings,
                    source_rows: Range::from(start - image.start..end - image.start),
                    target_row: start - batch.start,
                });
            }
        }
        overrides
    }

    pub fn mrope_positions(
        &self,
        first_position: u32,
        length: usize,
    ) -> (MRoPEPositions, u32) {
        let mut positions = Vec::with_capacity(length);
        let mut position = first_position;
        for image in self.images.iter() {
            for _ in positions.len()..image.start {
                positions.push([position; 3]);
                position += 1;
            }
            for row in 0..image.height {
                for column in 0..image.width {
                    positions.push([position, position + row, position + column]);
                }
            }
            position += image.width.max(image.height);
        }
        for _ in positions.len()..length {
            positions.push([position; 3]);
            position += 1;
        }

        let positions = MRoPEPositions {
            positions: positions.into_boxed_slice(),
        };
        (positions, position)
    }
}

#[derive(Debug, Error)]
pub enum VisionEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Vision layer error: {0}")]
    Layer(#[from] VisionLayerEncodeError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearEncodeError<B>),
    #[error("MLP error: {0}")]
    Mlp(#[from] MlpEncodeError<B>),
}

#[derive(Debug, Error)]
pub enum VisionPrefillError<B: Backend> {
    #[error("Vision encoder error: {0}")]
    Encode(#[from] VisionEncodeError<B>),
    #[error("Image pad token index {start} is out of order or outside the input of {length} tokens")]
    ImagePosition {
        start: usize,
        length: usize,
    },
    #[error(transparent)]
    InvalidImage(#[from] VisionImageError),
    #[error("Model has no vision encoder")]
    NoVisionEncoder,
}

impl<B: Backend> VisionModel<B> {
    pub fn prefill(
        &self,
        images: &[VisionImageInput],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<VisionPrefillResult<B>, VisionPrefillError<B>> {
        let mut prefilled_images = Vec::<VisionPrefilledImage<B>>::with_capacity(images.len());
        // Every earlier image shifts the later ones by its extra tokens, as `expand_tokens` does
        let mut offset = 0;
        for input in images {
            let (width, height) = self.merged_image_grid(&input.image)?;
            let embeddings = self.encode(&input.image, command_buffer)?;
            let prefilled_image = VisionPrefilledImage {
                embeddings,
                start: input.start + offset,
                width,
                height,
            };
            prefilled_images.push(prefilled_image);
            offset += (width * height) as usize - 1;
        }

        let result = VisionPrefillResult {
            images: prefilled_images.into_boxed_slice(),
        };
        Ok(result)
    }

    fn encode(
        &self,
        image: &VisionImage,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, VisionEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("vision encode");

        let encoder_config = &self.config.encoder_config;
        let patch_size = encoder_config.patch_size;
        let grid_width = image.width / patch_size;
        let grid_height = image.height / patch_size;
        let rows = grid_width * grid_height;
        let patch_dim = encoder_config.in_channels * encoder_config.temporal_patch_size * patch_size * patch_size;

        let pixels = command_buffer.allocate_constant_from_slice(&image.pixels).map_err(VisionEncodeError::Backend)?;
        let mut patches = command_buffer
            .allocate_scratch_for_shape(&[rows, patch_dim], self.data_type)
            .map_err(VisionEncodeError::Backend)?;

        self.patchify.encode(
            &pixels,
            &mut patches,
            &self.image_mean,
            &self.image_std,
            image.width,
            patch_size,
            encoder_config.spatial_merge_size,
            encoder_config.temporal_patch_size,
            rows * patch_dim,
            command_buffer,
        );

        let mut stream = self.patch_emb.encode(patches, rows, command_buffer)?;
        self.pos_emb.encode(&mut stream, grid_width, grid_height, encoder_config.spatial_merge_size, command_buffer);

        let rope = Self::vision_rope(
            &encoder_config.rope_config,
            grid_width,
            grid_height,
            encoder_config.spatial_merge_size,
            command_buffer,
        )
        .map_err(VisionEncodeError::Backend)?;

        for layer in self.layers.iter() {
            stream = layer.encode(stream, &rope, rows, command_buffer)?;
        }

        let normalized = self
            .merger_norm
            .encode(&stream, 0, rows, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(VisionEncodeError::Backend)?;

        let merged_rows = rows / encoder_config.spatial_merge_size.pow(2);
        let embeddings = self.merger_mlp.encode(normalized, merged_rows, command_buffer)?;

        Ok(embeddings)
    }

    fn vision_rope(
        rope_config: &VisionRoPEConfig,
        grid_width: u32,
        grid_height: u32,
        merge_size: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<PrecalculatedRoPE<B>, B::Error> {
        let dim = rope_config.head_dim as usize;
        let half_dim = dim / 2;
        let frequency_count = dim / 4;
        // inv_freq = 1 / base ^ (arange(0, half_dim, 2) / half_dim)
        let inverse_frequencies: Vec<f32> =
            (0..frequency_count).map(|j| 1.0 / rope_config.base.powf((2 * j) as f32 / half_dim as f32)).collect();

        let rows = (grid_width * grid_height) as usize;
        let mut cosines = vec![0.0; rows * dim];
        let mut sines = vec![0.0; rows * dim];
        for row in 0..rows {
            let (x, y) = vision_patch_coordinates(row as u32, grid_width, merge_size);
            for pair in 0..half_dim {
                // the first quarter of channels rotates by the row, the second by the column
                let coordinate = if pair < frequency_count {
                    y
                } else {
                    x
                };
                let angle = coordinate as f32 * inverse_frequencies[pair % frequency_count];
                let (sine, cosine) = angle.sin_cos();
                for channel in [row * dim + pair, row * dim + pair + half_dim] {
                    cosines[channel] = cosine;
                    sines[channel] = sine;
                }
            }
        }

        Ok(PrecalculatedRoPE {
            cosines: command_buffer.allocate_constant_from_slice(&cosines)?,
            sines: command_buffer.allocate_constant_from_slice(&sines)?,
            dim: dim as u32,
        })
    }
}
