use std::time::Instant;

use super::{allocation_stats::AllocationStats, allocator_trait::AllocatorTrait, model_config::ModelConfig};

const F16_SIZE: usize = 2;

#[derive(Clone, Copy, Debug)]
pub struct MemorySnapshot {
    pub step: usize,
    pub active_memory: usize,
    pub cache_memory: usize,
    pub peak_memory: usize,
}

pub struct InferenceSimulation<'a, A: AllocatorTrait> {
    allocator: &'a A,
    config: ModelConfig,
    pub stats: AllocationStats,
    pub current_prefix_len: usize,
    step_counter: usize,
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
            step_counter: 0,
        }
    }

    pub fn capture_snapshot(&self) -> MemorySnapshot {
        MemorySnapshot {
            step: self.step_counter,
            active_memory: self.allocator.active_memory(),
            cache_memory: self.allocator.cache_memory(),
            peak_memory: self.allocator.peak_memory(),
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

        let bias_size = batch_size * (batch_size + self.current_prefix_len) * F16_SIZE;
        buffers.push(self.alloc_scratch(bias_size));

        let rotated_q_size = self.config.num_heads * batch_size * self.config.head_dim * F16_SIZE;
        buffers.push(self.alloc_scratch(rotated_q_size));

        let rotated_k_size = self.config.num_groups * batch_size * self.config.head_dim * F16_SIZE;
        buffers.push(self.alloc_scratch(rotated_k_size));

        let attn_out_size = batch_size * self.config.num_heads * self.config.head_dim * F16_SIZE;
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
        self.step_counter += 1;

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
    ) -> Vec<MemorySnapshot> {
        let mut snapshots = Vec::new();
        let mut remaining = prompt_length;

        while remaining > 0 {
            let batch_size = remaining.min(step_size);
            self.simulate_forward_pass(batch_size);
            self.current_prefix_len += batch_size;
            remaining = remaining.saturating_sub(step_size);
            snapshots.push(self.capture_snapshot());
        }

        snapshots
    }

    pub fn run_generation(
        &mut self,
        num_tokens: usize,
    ) -> Vec<MemorySnapshot> {
        let mut snapshots = Vec::new();

        for _ in 0..num_tokens {
            self.simulate_forward_pass(1);
            self.current_prefix_len += 1;
            snapshots.push(self.capture_snapshot());
        }

        snapshots
    }

    pub fn run_generation_with_sampling(
        &mut self,
        num_tokens: usize,
        snapshot_interval: usize,
    ) -> Vec<MemorySnapshot> {
        let mut snapshots = Vec::new();

        for i in 0..num_tokens {
            self.simulate_forward_pass(1);
            self.current_prefix_len += 1;

            if i % snapshot_interval == 0 || i == num_tokens - 1 {
                snapshots.push(self.capture_snapshot());
            }
        }

        snapshots
    }
}
