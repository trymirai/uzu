use super::{
    allocation_stats::AllocationStats,
    allocator_trait::AllocatorTrait,
    inference_simulation::{InferenceSimulation, MemorySnapshot},
    model_config::ModelConfig,
};

pub struct SimulationResult {
    pub stats: AllocationStats,
    pub peak_memory_bytes: usize,
    pub cache_memory_bytes: usize,
    pub final_prefix_len: usize,
    pub snapshots: Vec<MemorySnapshot>,
}

pub fn run_simulation<A: AllocatorTrait>(
    allocator: &A,
    prompt_length: usize,
    generate_length: usize,
    prefill_step_size: usize,
) -> SimulationResult {
    let config = ModelConfig::llama_3_2_1b();
    let mut sim = InferenceSimulation::new(allocator, config);

    let mut snapshots = sim.run_prefill(prompt_length, prefill_step_size);
    let gen_snapshots = sim.run_generation(generate_length);
    snapshots.extend(gen_snapshots);

    SimulationResult {
        stats: sim.stats,
        peak_memory_bytes: allocator.peak_memory(),
        cache_memory_bytes: allocator.cache_memory(),
        final_prefix_len: sim.current_prefix_len,
        snapshots,
    }
}

pub fn run_long_simulation<A: AllocatorTrait>(
    allocator: &A,
    num_conversations: usize,
    avg_prompt_length: usize,
    avg_generate_length: usize,
    prefill_step_size: usize,
    snapshot_interval: usize,
) -> SimulationResult {
    let config = ModelConfig::llama_3_2_1b();
    run_long_simulation_with_config(
        allocator,
        config,
        num_conversations,
        avg_prompt_length,
        avg_generate_length,
        prefill_step_size,
        snapshot_interval,
    )
}

pub fn run_simulation_with_config<A: AllocatorTrait>(
    allocator: &A,
    config: ModelConfig,
    prompt_length: usize,
    generate_length: usize,
    prefill_step_size: usize,
) -> SimulationResult {
    let mut sim = InferenceSimulation::new(allocator, config);

    let mut snapshots = sim.run_prefill(prompt_length, prefill_step_size);
    let gen_snapshots = sim.run_generation(generate_length);
    snapshots.extend(gen_snapshots);

    SimulationResult {
        stats: sim.stats,
        peak_memory_bytes: allocator.peak_memory(),
        cache_memory_bytes: allocator.cache_memory(),
        final_prefix_len: sim.current_prefix_len,
        snapshots,
    }
}

pub fn run_long_simulation_with_config<A: AllocatorTrait>(
    allocator: &A,
    config: ModelConfig,
    num_conversations: usize,
    avg_prompt_length: usize,
    avg_generate_length: usize,
    prefill_step_size: usize,
    snapshot_interval: usize,
) -> SimulationResult {
    let mut sim = InferenceSimulation::new(allocator, config);
    let mut all_snapshots = Vec::new();

    for conv_idx in 0..num_conversations {
        let prompt_len =
            avg_prompt_length + (conv_idx % 3) * (avg_prompt_length / 4);
        let gen_len =
            avg_generate_length + (conv_idx % 5) * (avg_generate_length / 8);

        let prefill_snapshots = sim.run_prefill(prompt_len, prefill_step_size);
        all_snapshots.extend(prefill_snapshots);

        let gen_snapshots =
            sim.run_generation_with_sampling(gen_len, snapshot_interval);
        all_snapshots.extend(gen_snapshots);

        sim.current_prefix_len = 0;
    }

    SimulationResult {
        stats: sim.stats,
        peak_memory_bytes: allocator.peak_memory(),
        cache_memory_bytes: allocator.cache_memory(),
        final_prefix_len: sim.current_prefix_len,
        snapshots: all_snapshots,
    }
}
