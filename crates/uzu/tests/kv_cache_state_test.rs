#![cfg(any(target_os = "macos", target_os = "ios"))]

use metal::{MTLCommandBuffer, MTLCommandQueue};
use uzu::{
    DataType,
    array::ArrayContextExt,
    backends::{
        common::Context,
        metal::{KVCacheUpdate, MTLContext, Metal},
    },
    forward_pass::kv_cache_layer::{INVALID_POSITION, KVCacheLayer, KVCacheLayerState},
};

#[derive(Debug)]
struct Scenario {
    name: &'static str,
    state: KVCacheLayerState,
    prefix_capacity: usize,
    suffix_capacity: usize,
    initial_prefix_positions: Vec<usize>,
    accepted_suffix_indices: Vec<usize>,
    accepted_token_positions: Vec<usize>,
    suffix_token_positions: Vec<usize>,
    suffix_start: Option<usize>,
    expected_prefix_positions: Vec<usize>,
    expected_mask: Vec<Vec<bool>>,
    expected_ring_offset: Option<usize>,
    expected_ring_length: Option<usize>,
    expected_prefix_len: Option<usize>,
    expected_prefix_segment_length: usize,
}

fn make_test_layer(
    context: &MTLContext,
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
    let shape = [1, total_len.max(1), 1];

    let keys = std::cell::RefCell::new(context.create_array(&shape, DataType::F32, "kv_cache_keys"));
    let values = std::cell::RefCell::new(context.create_array(&shape, DataType::F32, "kv_cache_values"));

    let prefix_token_positions = match &state {
        KVCacheLayerState::Full {
            ..
        } => Vec::with_capacity(prefix_capacity),
        KVCacheLayerState::Windowed {
            window_length,
            ..
        } => vec![INVALID_POSITION; *window_length],
    };

    KVCacheLayer {
        state,
        keys,
        values,
        prefix_token_positions,
        max_suffix_length: suffix_capacity,
    }
}

fn fill_arrays(layer: &mut KVCacheLayer<Metal>) -> (Vec<f32>, Vec<f32>) {
    let initial_keys = {
        let mut keys_ref = layer.keys.borrow_mut();
        let slice = keys_ref.as_slice_mut::<f32>();
        for (idx, value) in slice.iter_mut().enumerate() {
            *value = 1_000.0 + idx as f32;
        }
        slice.to_vec()
    };

    let initial_values = {
        let mut values_ref = layer.values.borrow_mut();
        let slice = values_ref.as_slice_mut::<f32>();
        for (idx, value) in slice.iter_mut().enumerate() {
            *value = 2_000.0 + idx as f32;
        }
        slice.to_vec()
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

fn collect_mask(
    layer: &KVCacheLayer<Metal>,
    suffix_positions: &[usize],
) -> Vec<Vec<bool>> {
    let suffix_length = suffix_positions.len();
    let prefix_segment_length = layer.prefix_segment_length();

    (0..suffix_length)
        .map(|row| {
            (0..prefix_segment_length + suffix_length)
                .map(|col| layer.bias_should_be_neg_inf(row, col, suffix_positions))
                .collect()
        })
        .collect()
}

fn run_scenario(
    context: &MTLContext,
    scenario: &Scenario,
) {
    let mut layer =
        make_test_layer(context, scenario.state.clone(), scenario.prefix_capacity, scenario.suffix_capacity);

    layer.prefix_token_positions = scenario.initial_prefix_positions.clone();

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

    let mask = collect_mask(&layer, &scenario.suffix_token_positions);
    assert_eq!(mask, scenario.expected_mask, "{}: bias mask mismatch", scenario.name);

    let kv_cache_update = match KVCacheUpdate::new(context, DataType::F32, total_sequence_length) {
        Ok(update) => update,
        Err(e) => {
            panic!("Failed to create KV cache update for scenario {}: {:?}", scenario.name, e);
        },
    };

    let command_buffer = context.command_queue.command_buffer().expect("Failed to create command buffer").to_owned();

    let root_command_buffer = command_buffer.clone();
    layer.update_after_acceptance(
        &scenario.accepted_suffix_indices,
        scenario.suffix_start,
        &root_command_buffer,
        &kv_cache_update,
    );

    command_buffer.commit();
    root_command_buffer.wait_until_completed();

    layer.register_accepted_tokens(&scenario.accepted_token_positions);

    let actual_keys = {
        let keys_ref = layer.keys.borrow();
        keys_ref.as_slice::<f32>().to_vec()
    };
    assert_eq!(actual_keys, expected_keys, "{}: key buffer mismatch", scenario.name);

    let actual_values = {
        let values_ref = layer.values.borrow();
        values_ref.as_slice::<f32>().to_vec()
    };
    assert_eq!(actual_values, expected_values, "{}: value buffer mismatch", scenario.name);

    assert_eq!(
        layer.prefix_token_positions, scenario.expected_prefix_positions,
        "{}: prefix positions mismatch",
        scenario.name
    );

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
fn kv_cache_state_and_mask_scenarios() {
    let Some(context) = MTLContext::new().ok() else {
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
            initial_prefix_positions: vec![6, 1, 2, 3, 4, 5],
            accepted_suffix_indices: vec![0, 1, 2],
            accepted_token_positions: vec![7, 8, 9],
            suffix_token_positions: vec![7, 8, 9],
            suffix_start: None,
            expected_prefix_positions: vec![6, 7, 8, 9, 4, 5],
            expected_mask: vec![
                vec![false, true, false, false, false, false, false, true, true],
                vec![false, true, true, false, false, false, false, false, true],
                vec![false, true, true, true, false, false, false, false, false],
            ],
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
            initial_prefix_positions: vec![0, 1, 2, 3, INVALID_POSITION, INVALID_POSITION],
            accepted_suffix_indices: vec![0, 1, 2],
            accepted_token_positions: vec![4, 5, 6],
            suffix_token_positions: vec![4, 5, 6],
            suffix_start: None,
            expected_prefix_positions: vec![6, 1, 2, 3, 4, 5],
            expected_mask: vec![
                vec![false, false, false, false, true, true, false, true, true],
                vec![false, false, false, false, true, true, false, false, true],
                vec![true, false, false, false, true, true, false, false, false],
            ],
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
            initial_prefix_positions: vec![11, 12, 13, 14, 15, 10],
            accepted_suffix_indices: vec![0],
            accepted_token_positions: vec![16],
            suffix_token_positions: vec![16],
            suffix_start: None,
            expected_prefix_positions: vec![11, 12, 13, 14, 15, 16],
            expected_mask: vec![vec![false, false, false, false, false, true, false]],
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
            initial_prefix_positions: vec![0, 1, 2, 3],
            accepted_suffix_indices: vec![0, 1, 2],
            accepted_token_positions: vec![4, 5, 6],
            suffix_token_positions: vec![4, 5, 6],
            suffix_start: None,
            expected_prefix_positions: vec![0, 1, 2, 3, 4, 5, 6],
            expected_mask: vec![
                vec![false, false, false, false, false, true, true],
                vec![false, false, false, false, false, false, true],
                vec![false, false, false, false, false, false, false],
            ],
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
    let Some(context) = MTLContext::new().ok() else {
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
    layer.prefix_token_positions = vec![10, 11, 12, INVALID_POSITION];
    let (initial_keys, initial_values) = fill_arrays(&mut layer);

    let slice = layer.slice(&context, 0..2).expect("slice should exist");
    // Mutate the captured slots.
    {
        layer.prefix_token_positions[0] = 999;
        layer.prefix_token_positions[1] = 998;
        let mut keys = layer.keys.borrow_mut();
        let mut values = layer.values.borrow_mut();
        keys.as_slice_mut::<f32>()[0] = -1.0;
        keys.as_slice_mut::<f32>()[1] = -2.0;
        values.as_slice_mut::<f32>()[0] = -3.0;
        values.as_slice_mut::<f32>()[1] = -4.0;
    }

    layer.apply_slice(&slice, None);

    assert_eq!(
        layer.prefix_token_positions,
        vec![10, 11, 12, INVALID_POSITION],
        "positions restored for contiguous slice"
    );

    let keys_after = layer.keys.borrow().as_slice::<f32>().to_vec();
    let values_after = layer.values.borrow().as_slice::<f32>().to_vec();
    assert_eq!(keys_after[0..4], initial_keys[0..4], "keys restored for contiguous slice");
    assert_eq!(values_after[0..4], initial_values[0..4], "values restored for contiguous slice");
}

#[test]
fn kv_cache_slice_apply_wrap_window() {
    let Some(context) = MTLContext::new().ok() else {
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
    layer.prefix_token_positions = vec![30, 31, 32, 33];
    let (initial_keys, initial_values) = fill_arrays(&mut layer);

    let slice = layer.slice(&context, 2..4).expect("slice should exist");
    // Captured slots are expected to wrap; mutate them.
    {
        layer.prefix_token_positions[2] = 777;
        layer.prefix_token_positions[3] = 778;
        let mut keys = layer.keys.borrow_mut();
        let mut values = layer.values.borrow_mut();
        keys.as_slice_mut::<f32>()[2] = -11.0;
        keys.as_slice_mut::<f32>()[3] = -12.0;
        values.as_slice_mut::<f32>()[2] = -13.0;
        values.as_slice_mut::<f32>()[3] = -14.0;
    }

    layer.apply_slice(&slice, None);

    assert_eq!(layer.prefix_token_positions, vec![30, 31, 32, 33], "positions restored for wrapped slice");

    let keys_after = layer.keys.borrow().as_slice::<f32>().to_vec();
    let values_after = layer.values.borrow().as_slice::<f32>().to_vec();
    assert_eq!(keys_after[0..4], initial_keys[0..4], "keys restored for wrapped slice");
    assert_eq!(values_after[0..4], initial_values[0..4], "values restored for wrapped slice");
}

#[test]
fn kv_cache_slice_apply_full_restores_metadata() {
    let Some(context) = MTLContext::new().ok() else {
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
    layer.prefix_token_positions = vec![1, 2, 3];

    let slice = layer.slice(&context, 0..2).expect("full slice exists");

    // Mutate metadata.
    if let KVCacheLayerState::Full {
        prefix_len,
    } = &mut layer.state
    {
        *prefix_len = 1;
    }
    layer.prefix_token_positions = vec![9, 9, 9, 9];

    layer.apply_slice(&slice, None);

    assert_eq!(layer.prefix_token_positions, vec![1, 2, 3], "full slice restores positions");
    if let KVCacheLayerState::Full {
        prefix_len,
    } = &layer.state
    {
        assert_eq!(*prefix_len, 3, "full slice restores prefix_len");
    } else {
        panic!("expected full state");
    }
}
