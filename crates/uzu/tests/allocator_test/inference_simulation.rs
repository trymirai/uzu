use std::time::Instant;

use super::{AllocationStats, AllocatorTrait, ModelConfig};

const F16_SIZE: usize = 2;

pub struct InferenceSimulation<'a, A: AllocatorTrait> {
    allocator: &'a A,
    config: ModelConfig,
    pub stats: AllocationStats,
    pub current_prefix_len: usize,
}

impl<'a, A: AllocatorTrait> InferenceSimulation<'a, A> {
    pub fn new(
        allocator: &'a A,
        config: ModelConfig,
    ) -> Self {
        Self {
            allocator,
            config,
            stats: AllocationStats::default(),
            current_prefix_len: 0,
        }
    }

    fn alloc_scratch(
        &mut self,
        size: usize,
    ) -> A::Buffer {
        let start = Instant::now();
        let result = self.allocator.alloc(size);
        self.stats.record_alloc(size, start.elapsed());
        result
    }

    fn free(
        &mut self,
        buffer: A::Buffer,
    ) {
        let start = Instant::now();
        self.allocator.free(buffer);
        self.stats.record_free(start.elapsed());
    }

    fn simulate_embedding_kernel(
        &mut self,
        batch_size: usize,
    ) -> A::Buffer {
        let size = batch_size * self.config.model_dim * F16_SIZE;
        self.alloc_scratch(size)
    }

    fn simulate_attention_kernel(
        &mut self,
        batch_size: usize,
    ) -> Vec<A::Buffer> {
        let mut buffers = Vec::new();

        let qkv_size = batch_size * self.config.qkv_dim() * F16_SIZE;
        buffers.push(self.alloc_scratch(qkv_size));

        let bias_size =
            batch_size * (batch_size + self.current_prefix_len) * F16_SIZE;
        buffers.push(self.alloc_scratch(bias_size));

        let rotated_q_size = self.config.num_heads
            * batch_size
            * self.config.head_dim
            * F16_SIZE;
        buffers.push(self.alloc_scratch(rotated_q_size));

        let rotated_k_size = self.config.num_groups
            * batch_size
            * self.config.head_dim
            * F16_SIZE;
        buffers.push(self.alloc_scratch(rotated_k_size));

        let attn_out_size = batch_size
            * self.config.num_heads
            * self.config.head_dim
            * F16_SIZE;
        buffers.push(self.alloc_scratch(attn_out_size));

        buffers
    }

    fn simulate_mlp_kernel(
        &mut self,
        batch_size: usize,
    ) -> Vec<A::Buffer> {
        let mut buffers = Vec::new();

        let up_size = batch_size * 2 * self.config.hidden_dim * F16_SIZE;
        buffers.push(self.alloc_scratch(up_size));

        let hidden_size = batch_size * self.config.hidden_dim * F16_SIZE;
        buffers.push(self.alloc_scratch(hidden_size));

        buffers
    }

    fn simulate_logits_kernel(
        &mut self,
        batch_size: usize,
    ) -> A::Buffer {
        let size = batch_size * self.config.vocab_size * F16_SIZE;
        self.alloc_scratch(size)
    }

    fn simulate_forward_pass(
        &mut self,
        batch_size: usize,
    ) {
        let main = self.simulate_embedding_kernel(batch_size);

        for _layer in 0..self.config.num_layers {
            let attn_buffers = self.simulate_attention_kernel(batch_size);
            for buf in attn_buffers {
                self.free(buf);
            }

            let mlp_buffers = self.simulate_mlp_kernel(batch_size);
            for buf in mlp_buffers {
                self.free(buf);
            }
        }

        let logits = self.simulate_logits_kernel(batch_size);
        self.free(logits);
        self.free(main);
    }

    pub fn run_prefill(
        &mut self,
        prompt_length: usize,
        step_size: usize,
    ) {
        let mut remaining = prompt_length;

        while remaining > 0 {
            let batch_size = remaining.min(step_size);
            self.simulate_forward_pass(batch_size);
            self.current_prefix_len += batch_size;
            remaining = remaining.saturating_sub(step_size);
        }
    }

    pub fn run_generation(
        &mut self,
        num_tokens: usize,
    ) {
        for _ in 0..num_tokens {
            self.simulate_forward_pass(1);
            self.current_prefix_len += 1;
        }
    }
}
