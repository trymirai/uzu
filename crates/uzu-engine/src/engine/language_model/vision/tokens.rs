use shoji::types::session::vision::VisionImage;

use crate::{
    backends::Backend,
    engine::language_model::vision::{
        model::VisionModel,
        patch_grid::{VisionImageError, merged_image_grid},
    },
};

impl<B: Backend> VisionModel<B> {
    pub(super) fn merged_image_grid(
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

    pub fn image_token_count(
        &self,
        image: &VisionImage,
    ) -> Result<usize, VisionImageError> {
        let (token_columns, token_rows) = self.merged_image_grid(image)?;
        Ok((token_columns * token_rows) as usize)
    }
}
