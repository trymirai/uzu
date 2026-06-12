use monostate::MustBe;
use proc_macros::uzu_config;

#[uzu_config(super::EmbeddingConfig)]
pub struct UntiedEmbeddingConfig;

impl UntiedEmbeddingConfig {
    pub fn new(
        input_scale: Option<f32>,
        logit_soft_cap: Option<f32>,
    ) -> Self {
        Self {
            ty: MustBe!("UntiedEmbeddingConfig"),
            input_scale,
            logit_soft_cap,
        }
    }
}
