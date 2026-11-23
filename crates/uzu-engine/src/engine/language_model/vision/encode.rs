use shoji::types::session::vision::VisionImage;
use thiserror::Error;

use crate::{
    backends::{
        Backend,
        common::{CommandBuffer, CommandBufferEncoding, kernel::VisionPatchifyKernel},
    },
    config::vision::VisionRoPEConfig,
    encodable_block::{
        linear::LinearEncodeError, mixer::attention::rope::RoPEPositions, mlp::MlpEncodeError,
        vision_layer::VisionLayerEncodeError,
    },
    engine::language_model::vision::{
        model::VisionModel,
        patch_grid::{VisionImageError, vision_patch_coordinates},
    },
};

pub struct VisionPrefilledImage<B: Backend> {
    pub embeddings: B::ScratchBuffer,
    pub width: u32,
    pub height: u32,
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
    #[error("Input ends with an image, but generation continues from a token")]
    ImageAtEnd,
    #[error(transparent)]
    InvalidImage(#[from] VisionImageError),
    #[error("Model has no vision encoder")]
    NoVisionEncoder,
}

impl<B: Backend> VisionModel<B> {
    pub fn prefill(
        &self,
        image: &VisionImage,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<VisionPrefilledImage<B>, VisionPrefillError<B>> {
        let (width, height) = self.merged_image_grid(image)?;
        let embeddings = self.encode(image, command_buffer)?;

        Ok(VisionPrefilledImage {
            embeddings,
            width,
            height,
        })
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
    ) -> Result<RoPEPositions<B>, B::Error> {
        let frequency_count = rope_config.head_dim / 4;

        RoPEPositions::new(
            (grid_width * grid_height) as usize,
            rope_config.head_dim,
            |row, pair| {
                let (x, y) = vision_patch_coordinates(row as u32, grid_width, merge_size);
                // the first quarter of channels rotates by the row, the second by the column
                if pair < frequency_count {
                    y
                } else {
                    x
                }
            },
            command_buffer,
        )
    }
}
