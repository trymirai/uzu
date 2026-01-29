use super::{AllocationStats, AllocatorTrait, InferenceSimulation, ModelConfig};

pub struct SimulationResult {
    pub stats: AllocationStats,
    pub peak_memory_bytes: usize,
    pub cache_memory_bytes: usize,
    pub final_prefix_len: usize,
}

pub fn run_simulation<A: AllocatorTrait>(
    allocator: &A,
    prompt_length: usize,
    generate_length: usize,
    prefill_step_size: usize,
) -> SimulationResult {
    let config = ModelConfig::llama_3_2_1b();
    let mut sim = InferenceSimulation::new(allocator, config);

    sim.run_prefill(prompt_length, prefill_step_size);
    sim.run_generation(generate_length);

    SimulationResult {
        stats: sim.stats,
        peak_memory_bytes: allocator.peak_memory(),
        cache_memory_bytes: allocator.cache_memory(),
        final_prefix_len: sim.current_prefix_len,
    }
}
