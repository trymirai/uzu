#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsChunkPolicy {
    Fixed,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsNonStreamingMode {
    FullDecode,
    ChunkedIfNeeded,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TtsRunConfig {
    pub streaming_enabled: bool,
    pub target_emit_latency_ms: u32,
    pub initial_chunk_frames: usize,
    pub min_chunk_frames: usize,
    pub max_chunk_frames: usize,
    pub max_stream_workspace_frames: usize,
    pub max_semantic_frames: usize,
    pub chunk_policy: TtsChunkPolicy,
    pub non_streaming_mode: TtsNonStreamingMode,
    pub default_seed: u64,
    pub chunk_ema_alpha: f64,
    pub chunk_hysteresis_fraction: f64,
}

impl TtsRunConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.initial_chunk_frames == 0 {
            return Err("initial_chunk_frames must be greater than zero");
        }
        if self.min_chunk_frames == 0 {
            return Err("min_chunk_frames must be greater than zero");
        }
        if self.max_chunk_frames < self.min_chunk_frames {
            return Err("max_chunk_frames must be greater than or equal to min_chunk_frames");
        }
        if self.max_stream_workspace_frames == 0 {
            return Err("max_stream_workspace_frames must be greater than zero");
        }
        if self.target_emit_latency_ms == 0 {
            return Err("target_emit_latency_ms must be greater than zero");
        }
        if self.max_semantic_frames == 0 {
            return Err("max_semantic_frames must be greater than zero");
        }
        Ok(())
    }

    pub fn validate_stream_decode(&self) -> Result<(), &'static str> {
        self.validate()?;
        if self.max_stream_workspace_frames < self.max_chunk_frames {
            return Err("max_stream_workspace_frames must be greater than or equal to max_chunk_frames");
        }
        Ok(())
    }

    pub fn fixed_chunk_frames(chunk_frames: usize) -> Self {
        Self {
            chunk_policy: TtsChunkPolicy::Fixed,
            initial_chunk_frames: chunk_frames.max(1),
            min_chunk_frames: chunk_frames.max(1),
            max_chunk_frames: chunk_frames.max(1),
            ..Self::default()
        }
    }
}

impl Default for TtsRunConfig {
    fn default() -> Self {
        Self {
            streaming_enabled: true,
            target_emit_latency_ms: 80,
            initial_chunk_frames: 1,
            min_chunk_frames: 16,
            max_chunk_frames: 256,
            max_stream_workspace_frames: 768,
            max_semantic_frames: 768,
            chunk_policy: TtsChunkPolicy::Adaptive,
            non_streaming_mode: TtsNonStreamingMode::ChunkedIfNeeded,
            default_seed: 123,
            chunk_ema_alpha: 0.2,
            chunk_hysteresis_fraction: 0.25,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TtsPerformanceConfig {
    pub streaming: TtsRunConfig,
    pub non_streaming: TtsRunConfig,
    pub non_streaming_chunked_threshold_frames: usize,
}

impl Default for TtsPerformanceConfig {
    fn default() -> Self {
        let streaming = TtsRunConfig::default();
        let non_streaming = TtsRunConfig {
            streaming_enabled: false,
            chunk_policy: TtsChunkPolicy::Fixed,
            non_streaming_mode: TtsNonStreamingMode::ChunkedIfNeeded,
            initial_chunk_frames: 128,
            min_chunk_frames: 128,
            max_chunk_frames: 128,
            ..TtsRunConfig::default()
        };
        Self {
            streaming,
            non_streaming,
            non_streaming_chunked_threshold_frames: 4096,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TtsChunkPolicy, TtsNonStreamingMode, TtsPerformanceConfig, TtsRunConfig};

    #[test]
    fn default_run_config_matches_production_defaults() {
        let config = TtsRunConfig::default();
        assert!(config.streaming_enabled);
        assert_eq!(config.target_emit_latency_ms, 80);
        assert_eq!(config.initial_chunk_frames, 1);
        assert_eq!(config.min_chunk_frames, 16);
        assert_eq!(config.max_chunk_frames, 256);
        assert_eq!(config.max_stream_workspace_frames, 768);
        assert_eq!(config.max_semantic_frames, 768);
        assert_eq!(config.chunk_policy, TtsChunkPolicy::Adaptive);
        assert_eq!(config.non_streaming_mode, TtsNonStreamingMode::ChunkedIfNeeded);
        assert_eq!(config.default_seed, 123);
        assert!((config.chunk_ema_alpha - 0.2).abs() < f64::EPSILON);
        assert!((config.chunk_hysteresis_fraction - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn fixed_chunk_constructor_sets_bounds() {
        let config = TtsRunConfig::fixed_chunk_frames(64);
        assert_eq!(config.chunk_policy, TtsChunkPolicy::Fixed);
        assert_eq!(config.initial_chunk_frames, 64);
        assert_eq!(config.min_chunk_frames, 64);
        assert_eq!(config.max_chunk_frames, 64);
    }

    #[test]
    fn run_config_validation_rejects_invalid_bounds() {
        let invalid = TtsRunConfig {
            min_chunk_frames: 32,
            max_chunk_frames: 16,
            ..TtsRunConfig::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn run_config_validation_rejects_zero_initial_chunk() {
        let invalid = TtsRunConfig {
            initial_chunk_frames: 0,
            ..TtsRunConfig::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn run_config_validation_rejects_zero_semantic_cap() {
        let invalid = TtsRunConfig {
            max_semantic_frames: 0,
            ..TtsRunConfig::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn run_config_validation_rejects_zero_workspace_frames() {
        let invalid = TtsRunConfig {
            max_stream_workspace_frames: 0,
            ..TtsRunConfig::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn run_config_validation_rejects_workspace_smaller_than_max_chunk() {
        let invalid = TtsRunConfig {
            max_chunk_frames: 256,
            max_stream_workspace_frames: 128,
            ..TtsRunConfig::default()
        };
        assert!(invalid.validate_stream_decode().is_err());
    }

    #[test]
    fn performance_config_defaults_are_consistent() {
        let config = TtsPerformanceConfig::default();
        assert_eq!(config.streaming.chunk_policy, TtsChunkPolicy::Adaptive);
        assert_eq!(config.non_streaming.chunk_policy, TtsChunkPolicy::Fixed);
        assert_eq!(config.non_streaming.min_chunk_frames, config.non_streaming.max_chunk_frames);
        assert_eq!(config.non_streaming.non_streaming_mode, TtsNonStreamingMode::ChunkedIfNeeded);
    }
}
