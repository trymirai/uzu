#[derive(Clone, Copy)]
pub struct ModelConfig {
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
            model_dim: 2048,
            num_heads: 32,
            num_groups: 8,
            head_dim: 64,
            hidden_dim: 8192,
            vocab_size: 128256,
            num_layers: 16,
        }
    }

    pub fn qkv_dim(&self) -> usize {
        (2 * self.num_groups + self.num_heads) * self.head_dim
    }
}
