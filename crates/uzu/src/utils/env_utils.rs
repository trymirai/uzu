#[derive(Copy, Clone, Debug)]
pub enum EnvVar {
    CaptureFirstDecode,
    CaptureFirstPrefill,
    DisableAsyncGeneration,
    KvCompression,
    KvSparseValueKeepMass,
    KvSparseValuePageSize,
    KvSparseValueRecentWindow,
    KvShearKvValueBits,
    KvShapedKeyDims,
    KvShapedTarget,
    KvShapedValueDims,
    KvSpectralCalibration,
    KvSpectralTarget,
    KvSpectralValueBits,
    KvSpectralValueLayout,
    KvTriAttentionBudget,
    KvTriAttentionDivideLength,
    KvTurboQuantBits,
    KvTurboQuantTarget,
}

impl EnvVar {
    pub fn key(&self) -> &'static str {
        match self {
            EnvVar::CaptureFirstDecode => "UZU_CAPTURE_FIRST_DECODE",
            EnvVar::CaptureFirstPrefill => "UZU_CAPTURE_FIRST_PREFILL",
            EnvVar::DisableAsyncGeneration => "UZU_DISABLE_ASYNC_GENERATION",
            EnvVar::KvCompression => "UZU_KV_COMPRESSION",
            EnvVar::KvSparseValueKeepMass => "UZU_KV_SPARSE_VALUE_KEEP_MASS",
            EnvVar::KvSparseValuePageSize => "UZU_KV_SPARSE_VALUE_PAGE_SIZE",
            EnvVar::KvSparseValueRecentWindow => "UZU_KV_SPARSE_VALUE_RECENT_WINDOW",
            EnvVar::KvShearKvValueBits => "UZU_KV_SHEARKV_VALUE_BITS",
            EnvVar::KvShapedKeyDims => "UZU_KV_SHAPED_KEY_DIMS",
            EnvVar::KvShapedTarget => "UZU_KV_SHAPED_TARGET",
            EnvVar::KvShapedValueDims => "UZU_KV_SHAPED_VALUE_DIMS",
            EnvVar::KvSpectralCalibration => "UZU_KV_SPECTRAL_CALIBRATION",
            EnvVar::KvSpectralTarget => "UZU_KV_SPECTRAL_TARGET",
            EnvVar::KvSpectralValueBits => "UZU_KV_SPECTRAL_VALUE_BITS",
            EnvVar::KvSpectralValueLayout => "UZU_KV_SPECTRAL_VALUE_LAYOUT",
            EnvVar::KvTriAttentionBudget => "UZU_KV_TRIATTENTION_BUDGET",
            EnvVar::KvTriAttentionDivideLength => "UZU_KV_TRIATTENTION_DIVIDE_LENGTH",
            EnvVar::KvTurboQuantBits => "UZU_KV_TURBOQUANT_BITS",
            EnvVar::KvTurboQuantTarget => "UZU_KV_TURBOQUANT_TARGET",
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
