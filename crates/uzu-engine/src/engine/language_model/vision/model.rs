use std::{io, path::Path};

use thiserror::Error;

use crate::{
    backends::{
        Backend,
        common::{
            Context, Kernels,
            kernel::{TensorAddScaleKernel, VisionPatchifyKernel},
        },
    },
    config::vision::VisionConfig,
    data_type::DataType,
    encodable_block::{
        embedding_interpolated::InterpolatedPositionEmbedding,
        linear::{Linear, LinearBlockError},
        mlp::{UngatedMlp, UngatedMlpNewError},
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
        vision_layer::{VisionLayer, VisionLayerNewError},
    },
    safetensors::{ParameterLoader, ParameterLoaderError},
    utils::{buffer::create_buffer_with_data, file},
};

#[derive(Debug, Error)]
pub enum VisionLoadError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("Vision layer error: {0}")]
    Layer(#[from] VisionLayerNewError<B>),
    #[error("Linear block error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("ParameterLoader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("MLP error: {0}")]
    UngatedMlp(#[from] UngatedMlpNewError<B>),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported input channel count: {0}, expected RGB")]
    UnsupportedInChannels(u32),
}

pub struct VisionModel<B: Backend> {
    pub config: VisionConfig,
    pub(super) data_type: DataType,
    pub(super) image_mean: B::GlobalGpuBuffer,
    pub(super) image_std: B::GlobalGpuBuffer,
    pub(super) patchify: <B::Kernels as Kernels>::VisionPatchifyKernel,
    pub(super) patch_emb: Box<dyn Linear<B>>,
    pub(super) pos_emb: InterpolatedPositionEmbedding<B>,
    pub(super) pos_add: <B::Kernels as Kernels>::TensorAddScaleKernel,
    pub(super) layers: Box<[VisionLayer<B>]>,
    pub(super) merger_norm: Normalization<B>,
    pub(super) merger_mlp: UngatedMlp<B>,
}

impl<B: Backend> VisionModel<B> {
    pub async fn new(
        vision_path: &Path,
        data_type: DataType,
        context: &B::Context,
    ) -> Result<Self, VisionLoadError<B>> {
        if data_type != DataType::BF16 {
            return Err(VisionLoadError::UnsupportedDataType(data_type));
        };

        let config_file_content = &file::read(vision_path.join("config.json")).await?;
        let config = serde_json::from_slice::<VisionConfig>(config_file_content)?;
        // The patchify kernels take RGB pixels
        if config.encoder_config.in_channels != 3 {
            return Err(VisionLoadError::UnsupportedInChannels(config.encoder_config.in_channels));
        }

        let loader = ParameterLoader::new(vision_path.join("model.safetensors"), context).await?;
        let encoder_tree = loader.tree().subtree("encoder");

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
            &encoder_tree.subtree("patch_embedding"),
            context,
        )
        .await?;

        let num_positions = config.encoder_config.position_embedding_config.num_positions;
        let pos_emb = InterpolatedPositionEmbedding::new(
            data_type,
            num_positions,
            model_dim,
            &encoder_tree.subtree("position_embedding"),
        )
        .await?;

        let pos_add = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, true, true)
            .map_err(VisionLoadError::Backend)?;

        let mut layers = Vec::with_capacity(config.encoder_config.layer_configs.len());
        for (index, layer_config) in config.encoder_config.layer_configs.iter().enumerate() {
            let layer_tree = encoder_tree.subtree(&format!("layers.{index}"));
            let vision_layer = VisionLayer::new(
                data_type,
                layer_config,
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
            &encoder_tree.subtree("merger_norm"),
            context,
        )
        .await?;

        let merged_dim = model_dim * config.encoder_config.spatial_merge_size.pow(2);
        let merger_mlp = UngatedMlp::new(
            data_type,
            &config.encoder_config.merger_mlp_config,
            &encoder_tree.subtree("merger_mlp"),
            merged_dim,
            merged_dim,
            config.encoder_config.output_dim,
            context,
        )
        .await?;

        loader.assert_all_tensors_read()?;

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
            pos_add,
            layers: layers.into(),
            merger_norm,
            merger_mlp,
        })
    }
}
