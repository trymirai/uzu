use shoji::types::session::vision::VisionImage;
use thiserror::Error;

use crate::{
    backends::{
        Backend,
        common::{
            CommandBuffer, CommandBufferEncoding, Kernels, gpu_types::vision::CHANNELS, kernel::VisionPatchifyKernel,
        },
    },
    config::vision::VisionConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError, LinearEncodeError},
        mlp::{MlpEncodeError, UngatedMlp, UngatedMlpNewError},
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
        vision_layer::{VisionLayer, VisionLayerEncodeError, VisionLayerNewError},
    },
    engine::language_model::vision::{
        patch_grid::{VisionImageError, merged_image_grid},
        position_embedding::{VisionPositionEmbedding, VisionPositionEmbeddingNewError},
    },
    safetensors::ParameterTree,
    utils::buffer::create_buffer_with_data,
};

#[derive(Debug, Error)]
pub enum VisionLoadError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Vision layer error: {0}")]
    Layer(#[from] VisionLayerNewError<B>),
    #[error("Linear block error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Vision output dim {output_dim} doesn't match decoder model dim {decoder_model_dim}")]
    OutputDimMismatch {
        output_dim: u32,
        decoder_model_dim: u32,
    },
    #[error("Position embedding error: {0}")]
    PositionEmbedding(#[from] VisionPositionEmbeddingNewError<B>),
    #[error("MLP error: {0}")]
    UngatedMlp(#[from] UngatedMlpNewError<B>),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported input channel count: {0}, expected RGB")]
    UnsupportedInChannels(u32),
}

#[derive(Debug, Error)]
pub enum VisionEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Image error: {0}")]
    Image(#[from] VisionImageError),
    #[error("Input ends with an image, but generation continues from a token")]
    ImageAtEnd,
    #[error("Vision layer error: {0}")]
    Layer(#[from] VisionLayerEncodeError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearEncodeError<B>),
    #[error("MLP error: {0}")]
    Mlp(#[from] MlpEncodeError<B>),
    #[error("Model has no vision encoder")]
    NoVisionEncoder,
}

pub struct VisionEncodedImage<B: Backend> {
    pub embeddings: B::ScratchBuffer,
    pub width: u32,
    pub height: u32,
}

pub struct VisionModel<B: Backend> {
    pub config: VisionConfig,
    data_type: DataType,
    image_mean: B::GlobalGpuBuffer,
    image_std: B::GlobalGpuBuffer,
    patchify: <B::Kernels as Kernels>::VisionPatchifyKernel,
    patch_emb: Box<dyn Linear<B>>,
    pos_emb: VisionPositionEmbedding<B>,
    layers: Box<[VisionLayer<B>]>,
    merger_norm: Normalization<B>,
    merger_mlp: UngatedMlp<B>,
}

impl<B: Backend> VisionModel<B> {
    pub async fn new(
        data_type: DataType,
        config: VisionConfig,
        parameter_tree: &ParameterTree<'_, B>,
        decoder_model_dim: u32,
        context: &B::Context,
    ) -> Result<Self, VisionLoadError<B>> {
        if data_type != DataType::BF16 {
            return Err(VisionLoadError::UnsupportedDataType(data_type));
        };

        // The patchify kernels take RGB pixels
        if config.encoder_config.in_channels != CHANNELS {
            return Err(VisionLoadError::UnsupportedInChannels(config.encoder_config.in_channels));
        }

        // Image embeddings replace decoder token embeddings row by row
        if config.encoder_config.output_dim != decoder_model_dim {
            return Err(VisionLoadError::OutputDimMismatch {
                output_dim: config.encoder_config.output_dim,
                decoder_model_dim,
            });
        }

        let patchify =
            <B::Kernels as Kernels>::VisionPatchifyKernel::new(context, data_type).map_err(VisionLoadError::Backend)?;

        let patch_dim = config.encoder_config.in_channels
            * config.encoder_config.temporal_patch_size
            * config.encoder_config.patch_size.pow(2);
        let model_dim = config.encoder_config.model_dim;
        let patch_emb = <dyn Linear<B>>::new(
            patch_dim,
            model_dim,
            data_type,
            true,
            &parameter_tree.subtree("patch_embedding"),
            context,
        )
        .await?;

        let num_positions = config.encoder_config.position_embedding_config.num_positions;
        let pos_emb = VisionPositionEmbedding::new(
            data_type,
            num_positions,
            model_dim,
            &parameter_tree.subtree("position_embedding"),
            context,
        )
        .await?;

        let mut layers = Vec::with_capacity(config.encoder_config.layer_configs.len());
        for (index, layer_config) in config.encoder_config.layer_configs.iter().enumerate() {
            let layer_tree = parameter_tree.subtree(&format!("layers.{index}"));
            let vision_layer = VisionLayer::new(
                data_type,
                layer_config,
                &config.encoder_config.rope_config,
                &layer_tree,
                model_dim,
                config.encoder_config.hidden_dim,
                context,
            )
            .await?;
            layers.push(vision_layer);
        }

        let merger_norm = Normalization::new(
            model_dim,
            data_type,
            data_type,
            None,
            ShortcutMode::None,
            &config.encoder_config.merger_norm_config,
            &parameter_tree.subtree("merger_norm"),
            context,
        )
        .await?;

        let merged_dim = model_dim * config.encoder_config.spatial_merge_size.pow(2);
        let merger_mlp = UngatedMlp::new(
            data_type,
            &config.encoder_config.merger_mlp_config,
            &parameter_tree.subtree("merger_mlp"),
            merged_dim,
            merged_dim,
            config.encoder_config.output_dim,
            context,
        )
        .await?;

        let image_mean = create_buffer_with_data::<B, _>(context, &config.preprocessing_config.image_mean)
            .map_err(VisionLoadError::Backend)?;
        let image_std = create_buffer_with_data::<B, _>(context, &config.preprocessing_config.image_std)
            .map_err(VisionLoadError::Backend)?;

        Ok(Self {
            config,
            data_type,
            image_mean,
            image_std,
            patchify,
            patch_emb,
            pos_emb,
            layers: layers.into(),
            merger_norm,
            merger_mlp,
        })
    }

    pub fn encode(
        &self,
        image: &VisionImage,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<VisionEncodedImage<B>, VisionEncodeError<B>> {
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

        let batch_dim = BatchTopology::image(grid_width, grid_height, encoder_config.spatial_merge_size, 0..rows);
        for layer in self.layers.iter() {
            stream = layer.encode(stream, &batch_dim, command_buffer)?;
        }

        let normalized = self
            .merger_norm
            .encode(&stream, 0, rows, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(VisionEncodeError::Backend)?;

        let merged_rows = rows / encoder_config.spatial_merge_size.pow(2);
        let embeddings = self.merger_mlp.encode(normalized, merged_rows, command_buffer)?;

        let (width, height) = self.merged_image_grid(image)?;

        Ok(VisionEncodedImage {
            embeddings,
            width,
            height,
        })
    }

    pub fn image_token_count(
        &self,
        image: &VisionImage,
    ) -> Result<usize, VisionImageError> {
        let (token_columns, token_rows) = self.merged_image_grid(image)?;
        Ok((token_columns * token_rows) as usize)
    }

    fn merged_image_grid(
        &self,
        image: &VisionImage,
    ) -> Result<(u32, u32), VisionImageError> {
        let config = &self.config.encoder_config;
        let preprocessing_config = &self.config.preprocessing_config;
        merged_image_grid(
            image,
            config.patch_size,
            config.spatial_merge_size,
            config.temporal_patch_size,
            config.model_dim,
            preprocessing_config.min_pixels,
            preprocessing_config.max_pixels,
        )
    }
}
