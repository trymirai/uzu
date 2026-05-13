#![cfg(metal_backend)]

use crate::{
    DataType,
    backends::{
        common::{Allocation, Backend, Context, Encoder, kernel::kv_cache_update::KVCacheUpdate},
        metal::Metal,
    },
    forward_pass::kv_cache_layer::{KVCacheLayer, KVCacheLayerState},
};

#[path = "../../common/mod.rs"]
mod common;

use common::helpers::{alloc_allocation_with_data, allocation_to_vec, write_allocation};

#[derive(Debug)]
struct Scenario {
    name: &'static str,
    state: KVCacheLayerState,
    prefix_capacity: usize,
    suffix_capacity: usize,
    accepted_suffix_indices: Vec<usize>,
    number_of_accepted_tokens: usize,
    suffix_start: Option<usize>,
    expected_ring_offset: Option<usize>,
    expected_ring_length: Option<usize>,
    expected_prefix_len: Option<usize>,
    expected_prefix_segment_length: usize,
}

fn make_test_layer(
    context: &<Metal as Backend>::Context,
    state: KVCacheLayerState,
    prefix_capacity: usize,
    suffix_capacity: usize,
) -> KVCacheLayer<Metal> {
    let total_len = match &state {
        KVCacheLayerState::Full {
            ..
        } => prefix_capacity + suffix_capacity,
        KVCacheLayerState::Windowed {
            window_length,
            ..
        } => window_length + suffix_capacity,
    };
    let shape = [total_len.max(1), 1, 1];

    let zeroes = vec![0.0_f32; shape.iter().product()];
    let keys = alloc_allocation_with_data(context, &zeroes);
    let values = alloc_allocation_with_data(context, &zeroes);

    KVCacheLayer {
        state,
        keys,
        values,
        shape,
        data_type: DataType::F32,
    }
}

fn overwrite_allocation(
    allocation: &mut Allocation<Metal>,
    updates: &[(usize, f32)],
) {
    let mut data: Vec<f32> = allocation_to_vec(allocation);
    for (index, value) in updates {
        data[*index] = *value;
    }
    write_allocation(allocation, &data);
}

fn fill_arrays(layer: &mut KVCacheLayer<Metal>) -> (Vec<f32>, Vec<f32>) {
    let initial_keys = {
        let len = layer.shape.iter().product();
        let data: Vec<f32> = (0..len).map(|idx| 1_000.0 + idx as f32).collect();
        write_allocation(&mut layer.keys, &data);
        data
    };

    let initial_values = {
        let len = layer.shape.iter().product();
        let data: Vec<f32> = (0..len).map(|idx| 2_000.0 + idx as f32).collect();
        write_allocation(&mut layer.values, &data);
        data
    };

    (initial_keys, initial_values)
}

fn expected_after_update(
    state: &KVCacheLayerState,
    accepted_indices: &[usize],
    initial: &[f32],
) -> Vec<f32> {
    let mut expected = initial.to_vec();
    let effective_indices: Vec<usize> =
        if accepted_indices.is_empty() && matches!(state, KVCacheLayerState::Windowed { .. }) {
            vec![0]
        } else {
            accepted_indices.to_vec()
        };
    match state {
        KVCacheLayerState::Full {
            prefix_len,
        } => {
            let prefix_start = *prefix_len;
            for (offset, suffix_idx) in effective_indices.iter().enumerate() {
                let src = prefix_start + *suffix_idx;
                let dst = prefix_start + offset;
                if src != dst {
                    expected.swap(dst, src);
                }
            }
        },
        KVCacheLayerState::Windowed {
            ring_offset,
            ring_length,
            window_length,
        } => {
            for (offset, suffix_idx) in effective_indices.iter().enumerate() {
                let src = window_length + *suffix_idx;
                let dst = (ring_length + ring_offset + offset) % *window_length;
                expected.swap(dst, src);
            }
        },
    }
    expected
}

fn run_scenario(
    context: &<Metal as Backend>::Context,
    scenario: &Scenario,
) {
    let mut layer =
        make_test_layer(context, scenario.state.clone(), scenario.prefix_capacity, scenario.suffix_capacity);

    let state_before_update = layer.state.clone();

    let (initial_keys, initial_values) = fill_arrays(&mut layer);

    let expected_keys = expected_after_update(&state_before_update, &scenario.accepted_suffix_indices, &initial_keys);
    let expected_values =
        expected_after_update(&state_before_update, &scenario.accepted_suffix_indices, &initial_values);

    let total_sequence_length = match &layer.state {
        KVCacheLayerState::Full {
            ..
        } => scenario.prefix_capacity + scenario.suffix_capacity,
        KVCacheLayerState::Windowed {
            window_length,
            ..
        } => window_length + scenario.suffix_capacity,
    };

    let kv_cache_update = match KVCacheUpdate::new(context, DataType::F32, total_sequence_length) {
        Ok(update) => update,
        Err(e) => {
            panic!("Failed to create KV cache update for scenario {}: {:?}", scenario.name, e);
        },
    };

    let mut encoder = Encoder::new(context).unwrap();
    layer.update_after_acceptance(
        &scenario.accepted_suffix_indices,
        scenario.suffix_start,
        &mut encoder,
        &kv_cache_update,
    );

    encoder.end_encoding().submit().wait_until_completed().unwrap();

    layer.register_accepted_tokens(scenario.number_of_accepted_tokens);

    let actual_keys: Vec<f32> = allocation_to_vec(&layer.keys);
    assert_eq!(actual_keys, expected_keys, "{}: key buffer mismatch", scenario.name);

    let actual_values: Vec<f32> = allocation_to_vec(&layer.values);
    assert_eq!(actual_values, expected_values, "{}: value buffer mismatch", scenario.name);

    match &layer.state {
        KVCacheLayerState::Full {
            prefix_len,
        } => {
            if let Some(expected_len) = scenario.expected_prefix_len {
                assert_eq!(*prefix_len, expected_len, "{}: prefix length mismatch", scenario.name);
            }
        },
        KVCacheLayerState::Windowed {
            ring_offset,
            ring_length,
            ..
        } => {
            if let Some(expected_offset) = scenario.expected_ring_offset {
                assert_eq!(*ring_offset, expected_offset, "{}: ring offset mismatch", scenario.name);
            }
            if let Some(expected_length) = scenario.expected_ring_length {
                assert_eq!(*ring_length, expected_length, "{}: ring length mismatch", scenario.name);
            }
        },
    }

    assert_eq!(
        layer.prefix_segment_length(),
        scenario.expected_prefix_segment_length,
        "{}: prefix segment length mismatch",
        scenario.name
    );
}

#[test]
fn kv_cache_state_scenarios() {
    let Some(context) = <Metal as Backend>::Context::new().ok() else {
        return;
    };

    let scenarios = vec![
        Scenario {
            name: "windowed_after_crossing_window",
            state: KVCacheLayerState::Windowed {
                ring_offset: 1,
                ring_length: 6,
                window_length: 6,
            },
            prefix_capacity: 6,
            suffix_capacity: 3,
            accepted_suffix_indices: vec![0, 1, 2],
            number_of_accepted_tokens: 3,
            suffix_start: None,
            expected_ring_offset: Some(4),
            expected_ring_length: Some(6),
            expected_prefix_len: None,
            expected_prefix_segment_length: 6,
        },
        Scenario {
            name: "windowed_during_crossing_window",
            state: KVCacheLayerState::Windowed {
                ring_offset: 0,
                ring_length: 4,
                window_length: 6,
            },
            prefix_capacity: 6,
            suffix_capacity: 3,
            accepted_suffix_indices: vec![0, 1, 2],
            number_of_accepted_tokens: 3,
            suffix_start: None,
            expected_ring_offset: Some(1),
            expected_ring_length: Some(6),
            expected_prefix_len: None,
            expected_prefix_segment_length: 6,
        },
        Scenario {
            name: "windowed_single_accept_second_wrap",
            state: KVCacheLayerState::Windowed {
                ring_offset: 5,
                ring_length: 6,
                window_length: 6,
            },
            prefix_capacity: 6,
            suffix_capacity: 1,
            accepted_suffix_indices: vec![0],
            number_of_accepted_tokens: 1,
            suffix_start: None,
            expected_ring_offset: Some(0),
            expected_ring_length: Some(6),
            expected_prefix_len: None,
            expected_prefix_segment_length: 6,
        },
        Scenario {
            name: "full_mode_pure_causal",
            state: KVCacheLayerState::Full {
                prefix_len: 4,
            },
            prefix_capacity: 6,
            suffix_capacity: 3,
            accepted_suffix_indices: vec![0, 1, 2],
            number_of_accepted_tokens: 3,
            suffix_start: None,
            expected_ring_offset: None,
            expected_ring_length: None,
            expected_prefix_len: Some(7),
            expected_prefix_segment_length: 7,
        },
    ];

    for scenario in &scenarios {
        run_scenario(&context, scenario);
    }
}

#[test]
fn kv_cache_slice_apply_contiguous_window() {
    let Some(context) = <Metal as Backend>::Context::new().ok() else {
        return;
    };

    let mut layer = make_test_layer(
        &context,
        KVCacheLayerState::Windowed {
            ring_offset: 1,
            ring_length: 3,
            window_length: 4,
        },
        4,
        0,
    );
    let (initial_keys, initial_values) = fill_arrays(&mut layer);

    let slice = layer.slice(&context, 0..2).expect("slice should exist");
    // Mutate the captured slots.
    overwrite_allocation(&mut layer.keys, &[(0, -1.0), (1, -2.0)]);
    overwrite_allocation(&mut layer.values, &[(0, -3.0), (1, -4.0)]);

    layer.apply_slice(&slice, None);

    let keys_after: Vec<f32> = allocation_to_vec(&layer.keys);
    let values_after: Vec<f32> = allocation_to_vec(&layer.values);
    assert_eq!(keys_after[0..4], initial_keys[0..4], "keys restored for contiguous slice");
    assert_eq!(values_after[0..4], initial_values[0..4], "values restored for contiguous slice");
}

#[test]
fn kv_cache_slice_apply_wrap_window() {
    let Some(context) = <Metal as Backend>::Context::new().ok() else {
        return;
    };

    let mut layer = make_test_layer(
        &context,
        KVCacheLayerState::Windowed {
            ring_offset: 3,
            ring_length: 3,
            window_length: 4,
        },
        4,
        0,
    );
    let (initial_keys, initial_values) = fill_arrays(&mut layer);

    let slice = layer.slice(&context, 2..4).expect("slice should exist");
    // Captured slots are expected to wrap; mutate them.
    overwrite_allocation(&mut layer.keys, &[(2, -11.0), (3, -12.0)]);
    overwrite_allocation(&mut layer.values, &[(2, -13.0), (3, -14.0)]);

    layer.apply_slice(&slice, None);

    let keys_after: Vec<f32> = allocation_to_vec(&layer.keys);
    let values_after: Vec<f32> = allocation_to_vec(&layer.values);
    assert_eq!(keys_after[0..4], initial_keys[0..4], "keys restored for wrapped slice");
    assert_eq!(values_after[0..4], initial_values[0..4], "values restored for wrapped slice");
}

#[test]
fn kv_cache_slice_apply_full_restores_metadata() {
    let Some(context) = <Metal as Backend>::Context::new().ok() else {
        return;
    };

    let mut layer = make_test_layer(
        &context,
        KVCacheLayerState::Full {
            prefix_len: 3,
        },
        6,
        0,
    );

    let slice = layer.slice(&context, 0..2).expect("full slice exists");

    // Mutate metadata.
    if let KVCacheLayerState::Full {
        prefix_len,
    } = &mut layer.state
    {
        *prefix_len = 1;
    }

    layer.apply_slice(&slice, None);

    if let KVCacheLayerState::Full {
        prefix_len,
    } = &layer.state
    {
        assert_eq!(*prefix_len, 3, "full slice restores prefix_len");
    } else {
        panic!("expected full state");
    }
}
