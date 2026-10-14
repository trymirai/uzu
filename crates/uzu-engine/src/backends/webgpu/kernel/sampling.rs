use super::{NamedBinding, Param, Shader};
use crate::{
    backends::{
        common::CommandBufferEncoding,
        webgpu::command_buffer::{Binding, WebGpuCommandBufferEncoding},
    },
    data_type::DataType,
};

// Four contiguous 256-lane reads give enough independent groups to occupy the
// device even for smaller vocabularies. Slang also builds a 16-read variant for
// controlled tuning without changing the kernel contract.
const GREEDY_ITEMS_PER_LANE: u32 = 4;

// Small top-k policies keep only a bounded sorted list per 1024-token tile.
// Merge records include normalization over ALL logits, before top-k filtering.
fn dispatch_top_k(
    shader: &Shader,
    command: &mut WebGpuCommandBufferEncoding,
) -> bool {
    let k = shader.u("top_k");
    let mut groups = shader.u("vocab_size").div_ceil(1024);
    let batch = shader.u("batch_size");
    if !shader.flag("has_top_k") || !(1..=64).contains(&k) || !(1..=65_535).contains(&groups) || batch == 0 {
        return false;
    }
    let width = k.next_power_of_two();
    let record_words = width * 2 + 4;
    let Some(bytes) = (groups as usize)
        .checked_mul(batch as usize)
        .and_then(|n| n.checked_mul(record_words as usize * size_of::<u32>()))
    else {
        return false;
    };
    let Ok(mut current) = command.allocate_scratch(bytes) else {
        return false;
    };
    let Ok(mut next) = command.allocate_scratch(bytes) else {
        return false;
    };
    let mut base = shader.clone();
    let mut first = base.clone();
    first.specs.insert("topk_merge", Param::bool(false));
    first.specs.insert("write_final", Param::bool(groups == 1));
    first.set("topk_input_groups", 0);
    first.set("topk_output_groups", groups);
    first.bind(NamedBinding::new("topk_out", Binding::write(&mut current), DataType::U32));
    first.groups = [groups, batch, 1];
    first.kernel("sampling_topk", width);
    first.dispatch(command);
    while groups > 1 {
        let output_groups = groups.div_ceil(1024 / width);
        let mut merge = base.clone();
        merge.specs.insert("topk_merge", Param::bool(true));
        merge.specs.insert("write_final", Param::bool(output_groups == 1));
        merge.set("topk_input_groups", groups);
        merge.set("topk_output_groups", output_groups);
        merge.bind(NamedBinding::new("topk_in", Binding::read(&current), DataType::U32));
        merge.bind(NamedBinding::new("topk_out", Binding::write(&mut next), DataType::U32));
        merge.groups = [output_groups, batch, 1];
        merge.kernel("sampling_topk", width);
        merge.dispatch(command);
        std::mem::swap(&mut current, &mut next);
        groups = output_groups;
    }
    // A tiny positive temperature can overflow otherwise finite logits. Keep
    // the existing sampler's behavior for that exceptional row, entirely on GPU.
    base.specs.insert("topk_fallback", Param::bool(true));
    base.set("topk_record_words", record_words);
    base.bind(NamedBinding::new("topk_status", Binding::read(&current), DataType::U32));
    base.groups = [batch, 1, 1];
    base.kernel("sampling", 0);
    base.dispatch(command);
    true
}

pub(super) fn dispatch(
    name: &str,
    mut shader: Shader,
    command: &mut WebGpuCommandBufferEncoding,
) {
    if name == "UnifiedSampling"
        && !["is_stochastic", "has_temperature", "has_top_k", "has_top_p", "has_min_p"]
            .iter()
            .any(|flag| shader.flag(flag))
    {
        let partial_count = shader.u("vocab_size").div_ceil(256 * GREEDY_ITEMS_PER_LANE);
        let batch_size = shader.u("batch_size");
        if partial_count > 1 && partial_count <= 65_535 && batch_size > 0 {
            // Scratch comes from the command's existing allocation pool. It is
            // GPU-only and is reused after the ordered dispatches consume it.
            let size = (partial_count as usize)
                .checked_mul(batch_size as usize)
                .and_then(|count| count.checked_mul(2 * size_of::<u32>()));
            if let Some(size) = size
                && let Ok(mut partials) = command.allocate_scratch(size)
            {
                shader.set("partial_count", partial_count);
                let mut first = shader.clone();
                first.specs.insert("write_final", Param::bool(false));
                first.bind(NamedBinding::new("partials", Binding::write(&mut partials), DataType::U32));
                first.groups = [partial_count, batch_size, 1];
                first.kernel("sampling_greedy_partials", GREEDY_ITEMS_PER_LANE);
                first.dispatch(command);

                shader.bind(NamedBinding::new("partials", Binding::read(&partials), DataType::U32));
                shader.groups = [batch_size, 1, 1];
                shader.kernel("sampling_greedy_finish", 0);
                shader.dispatch(command);
                return;
            }
        }
        if partial_count > 0 {
            // Small vocabularies need no scratch. Allocation pressure or a
            // vocabulary beyond the dispatch limit uses the identical argmax
            // semantics with a longer single-workgroup scan.
            shader.specs.insert("write_final", Param::bool(true));
            shader.groups = [1, batch_size, 1];
            shader.kernel("sampling_greedy_partials", GREEDY_ITEMS_PER_LANE);
            shader.dispatch(command);
            return;
        }
    }
    if name == "UnifiedSampling" && dispatch_top_k(&shader, command) {
        return;
    }
    let kernel = match name {
        "UnifiedSampling" => {
            shader.groups = [shader.u("batch_size"), 1, 1];
            "sampling"
        },
        "RepetitionPenalty" => {
            shader.groups = [shader.u("suffix_repetition_length").div_ceil(256), shader.u("sampling_length"), 1];
            "repetition_penalty"
        },
        "ContextRingUpdate" => {
            shader.groups = [1, 1, 1];
            "context_ring_update"
        },
        _ => unreachable!(),
    };
    shader.kernel(kernel, 0);
    shader.dispatch(command);
}
