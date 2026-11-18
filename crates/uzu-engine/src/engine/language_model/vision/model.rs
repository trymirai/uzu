use std::{io, path::Path};

use derive_more::Debug;
use half::bf16;
use thiserror::Error;

use super::{
    layer::{VisionLayer, VisionLayerEncodeError, VisionLayerNewError},
    position_embedding::{InterpolatedPositionEmbedding, PositionEmbedding},
    rope::get_precalculated_vision_rope,
};
use crate::{
    backends::{
        Backend,
        common::{CommandBuffer, CommandBufferEncoding, Kernels, kernel::TensorAddScaleKernel},
    },
    config::vision::VisionConfig,
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError, LinearEncodeError},
        mlp::{Mlp, MlpEncodeError, UngatedMlp, UngatedMlpNewError},
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
    },
    safetensors::{ParameterLoader, ParameterLoaderError},
    utils::file,
};

#[derive(Debug, Error)]
pub enum VisionModelLoadError<B: Backend> {
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
}

#[derive(Debug, Error)]
pub enum VisionModelEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Vision layer error: {0}")]
    Layer(#[from] VisionLayerEncodeError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearEncodeError<B>),
    #[error("MLP error: {0}")]
    Mlp(#[from] MlpEncodeError<B>),
}

pub struct VisionModel<B: Backend> {
    config: VisionConfig,
    data_type: DataType,
    patch_embedding: Box<dyn Linear<B>>,
    position_embedding: PositionEmbedding,
    position_add: <B::Kernels as Kernels>::TensorAddScaleKernel,
    layers: Box<[VisionLayer<B>]>,
    merger_norm: Normalization<B>,
    merger_mlp: UngatedMlp<B>,
}

impl<B: Backend> VisionModel<B> {
    pub async fn new(
        vision_path: &Path,
        data_type: DataType,
        context: &B::Context,
    ) -> Result<Self, VisionModelLoadError<B>> {
        let config_file_content = &file::read(vision_path.join("config.json")).await?;
        let config: VisionConfig = serde_json::from_slice(config_file_content)?;

        let loader = ParameterLoader::new(vision_path.join("model.safetensors"), context).await?;
        let tree = loader.tree().subtree("encoder");

        let model_dim = config.encoder_config.model_dim;
        let patch_dim = config.encoder_config.in_channels
            * config.encoder_config.temporal_patch_size
            * config.encoder_config.patch_size.pow(2);

        let patch_embedding =
            <dyn Linear<B>>::new(patch_dim, model_dim, data_type, true, &tree.subtree("patch_embedding"), context)
                .await?;
        let num_positions = config.encoder_config.position_embedding_config.num_positions;
        let position_tree = tree.subtree("position_embedding");
        let position_embedding = match data_type {
            DataType::BF16 => PositionEmbedding::BF16(
                InterpolatedPositionEmbedding::load(num_positions, model_dim, &position_tree).await?,
            ),
            DataType::F32 => PositionEmbedding::F32(
                InterpolatedPositionEmbedding::load(num_positions, model_dim, &position_tree).await?,
            ),
            data_type => return Err(VisionModelLoadError::UnsupportedDataType(data_type)),
        };
        let position_add = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, true, true)
            .map_err(VisionModelLoadError::Backend)?;

        let merged_dim = model_dim * config.encoder_config.spatial_merge_size.pow(2);
        let mut layers = Vec::with_capacity(config.encoder_config.layer_configs.len());
        for (index, layer_config) in config.encoder_config.layer_configs.iter().enumerate() {
            let layer_tree = tree.subtree(&format!("layers.{index}"));
            layers.push(
                VisionLayer::new(
                    model_dim,
                    config.encoder_config.hidden_dim,
                    data_type,
                    layer_config,
                    &layer_tree,
                    context,
                )
                .await?,
            );
        }

        let merger_norm = Normalization::new(
            model_dim,
            data_type,
            data_type,
            None,
            ShortcutMode::None,
            &config.encoder_config.merger_norm_config,
            &tree.subtree("merger_norm"),
            context,
        )
        .await?;

        let merger_mlp = UngatedMlp::new(
            merged_dim,
            merged_dim,
            config.encoder_config.output_dim,
            data_type,
            &config.encoder_config.merger_mlp_config,
            &tree.subtree("merger_mlp"),
            context,
        )
        .await?;

        loader.assert_all_tensors_read()?;

        Ok(Self {
            config,
            data_type,
            patch_embedding,
            position_embedding,
            position_add,
            layers: layers.into(),
            merger_norm,
            merger_mlp,
        })
    }

    /// Encodes one image (a single frame) of `height x width` patches into `[height * width / (merge * merge), output_dim]` embeddings.
    /// `patches` is the preprocessor output: `[height * width, in_channels * temporal_patch_size * patch_size * patch_size]`
    /// in HF's patch order, where each `merge x merge` block of patches is contiguous.
    pub(super) fn encode(
        &self,
        patches: &[f32],
        height: u32,
        width: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, VisionModelEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("vision encoder");
        let encoder_config = &self.config.encoder_config;
        let patch_dim =
            encoder_config.in_channels * encoder_config.temporal_patch_size * encoder_config.patch_size.pow(2);
        let rows = height * width;
        assert_eq!(patches.len(), (rows * patch_dim) as usize, "expected {rows} patches of {patch_dim} values");

        let pixels = create_const_buffer::<B>(patches, self.data_type, command_buffer)
            .map_err(VisionModelEncodeError::Backend)?;
        let mut input = command_buffer
            .allocate_scratch_for_shape(&[rows, patch_dim], self.data_type)
            .map_err(VisionModelEncodeError::Backend)?;
        command_buffer.encode_copy(&pixels, &mut input);
        let mut stream_buffer = self.patch_embedding.encode(input, rows, command_buffer)?;

        let merge = encoder_config.spatial_merge_size;
        let positions = match &self.position_embedding {
            PositionEmbedding::BF16(embedding) => {
                command_buffer.allocate_constant_from_slice(&embedding.interpolate(height, width, merge))
            },
            PositionEmbedding::F32(embedding) => {
                command_buffer.allocate_constant_from_slice(&embedding.interpolate(height, width, merge))
            },
        }
        .map_err(VisionModelEncodeError::Backend)?;
        let elements = rows * encoder_config.model_dim;
        self.position_add.encode(
            None::<&B::ScratchBuffer>,
            Some(&positions),
            &mut stream_buffer,
            elements,
            elements,
            1.0,
            command_buffer,
        );

        let rope = get_precalculated_vision_rope(
            &encoder_config.rope_config,
            height,
            width,
            encoder_config.spatial_merge_size,
            command_buffer,
        )
        .map_err(VisionModelEncodeError::Backend)?;

        for layer in &self.layers {
            stream_buffer = layer.encode(stream_buffer, &rope, rows, command_buffer)?;
        }

        let normalized = self
            .merger_norm
            .encode(&stream_buffer, 0, rows, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(VisionModelEncodeError::Backend)?;

        // Each merge block's patches are adjacent rows, so the normalized rows already form the merged rows.
        let result =
            self.merger_mlp.encode(normalized, rows / encoder_config.spatial_merge_size.pow(2), command_buffer)?;
        Ok(result)
    }

    pub fn config(&self) -> &VisionConfig {
        &self.config
    }
}

fn create_const_buffer<B: Backend>(
    values: &[f32],
    data_type: DataType,
    command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
) -> Result<B::ConstantBuffer, B::Error> {
    match data_type {
        DataType::BF16 => command_buffer
            .allocate_constant_from_slice(&values.iter().map(|&value| bf16::from_f32(value)).collect::<Vec<_>>()),
        DataType::F32 => command_buffer.allocate_constant_from_slice(values),
        data_type => unreachable!("vision data type {data_type:?} is rejected at load"),
    }
}
