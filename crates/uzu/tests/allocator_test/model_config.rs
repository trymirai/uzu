#[derive(Clone, Copy)]
pub struct ModelConfig {
    pub name: &'static str,
    pub model_dim: usize,
    pub num_heads: usize,
    pub num_groups: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
}

impl ModelConfig {
    pub fn llama_3_2_1b() -> Self {
        Self {
            name: "Llama-3.2-1B",
            model_dim: 2048,
            num_heads: 32,
            num_groups: 8,
            head_dim: 64,
            hidden_dim: 8192,
            vocab_size: 128256,
            num_layers: 16,
        }
    }

    pub fn llama_3_2_3b() -> Self {
        Self {
            name: "Llama-3.2-3B",
            model_dim: 3072,
            num_heads: 24,
            num_groups: 8,
            head_dim: 128,
            hidden_dim: 8192,
            vocab_size: 128256,
            num_layers: 28,
        }
    }

    pub fn qwen3_4b() -> Self {
        Self {
            name: "Qwen3-4B",
            model_dim: 2560,
            num_heads: 32,
            num_groups: 8,
            head_dim: 128,
            hidden_dim: 9728,
            vocab_size: 151936,
            num_layers: 36,
        }
    }

    pub fn qwen3_8b() -> Self {
        Self {
            name: "Qwen3-8B",
            model_dim: 4096,
            num_heads: 32,
            num_groups: 8,
            head_dim: 128,
            hidden_dim: 12288,
            vocab_size: 151936,
            num_layers: 36,
        }
    }

    pub fn gemma_3_4b() -> Self {
        Self {
            name: "Gemma-3-4B",
            model_dim: 2560,
            num_heads: 8,
            num_groups: 4,
            head_dim: 256,
            hidden_dim: 10240,
            vocab_size: 262208,
            num_layers: 34,
        }
    }

    pub fn qwen3_14b() -> Self {
        Self {
            name: "Qwen3-14B",
            model_dim: 5120,
            num_heads: 40,
            num_groups: 8,
            head_dim: 128,
            hidden_dim: 17408,
            vocab_size: 151936,
            num_layers: 40,
        }
    }

    pub fn all_models() -> Vec<Self> {
        vec![
            Self::llama_3_2_1b(),
            Self::llama_3_2_3b(),
            Self::qwen3_4b(),
            Self::qwen3_8b(),
            Self::gemma_3_4b(),
            Self::qwen3_14b(),
        ]
    }

    pub fn qkv_dim(&self) -> usize {
        (2 * self.num_groups + self.num_heads) * self.head_dim
    }
}
