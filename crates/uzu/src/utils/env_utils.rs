#[derive(Copy, Clone, Debug)]
pub enum EnvVar {
    CaptureFirstDecode,
    CaptureFirstPrefill,
    MlpBlocksByLayerJson,
    MlpStaticBlocks,
    MlpStaticKeepRatio,
}

impl EnvVar {
    pub fn key(&self) -> &'static str {
        match self {
            EnvVar::CaptureFirstDecode => "UZU_CAPTURE_FIRST_DECODE",
            EnvVar::CaptureFirstPrefill => "UZU_CAPTURE_FIRST_PREFILL",
            EnvVar::MlpBlocksByLayerJson => "UZU_MLP_BLOCKS_BY_LAYER_JSON",
            EnvVar::MlpStaticBlocks => "UZU_MLP_STATIC_BLOCKS",
            EnvVar::MlpStaticKeepRatio => "UZU_MLP_STATIC_KEEP_RATIO",
        }
    }

    pub fn value(&self) -> String {
        std::env::var(self.key()).unwrap_or_default()
    }

    pub fn is_enabled(&self) -> bool {
        let upper = self.value().to_ascii_uppercase();
        matches!(upper.as_str(), "1" | "YES" | "TRUE")
    }
}
