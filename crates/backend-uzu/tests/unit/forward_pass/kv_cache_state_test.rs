#![cfg(metal_backend)]

use crate::{
    DataType,
    backends::{
        common::{Backend, Context, Encoder, kernel::kv_cache_update::KVCacheUpdate},
        metal::Metal,
    },
    forward_pass::kv_cache_layer::{KVCacheLayer, KVCacheLayerState},
    prelude::MetalContext,
    utils::SparseArrayContext,
};

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

    let keys = context.create_sparse_array(&shape, DataType::F32, "kv_cache_keys");
    let values = context.create_sparse_array(&shape, DataType::F32, "kv_cache_values");
    KVCacheLayer {
        state,
        keys,
        values,
    }
}

fn fill_arrays(
    context: &MetalContext,
    layer: &mut KVCacheLayer<Metal>,
) -> (Vec<f32>, Vec<f32>) {
    let element_count = layer.keys.shape().iter().product::<usize>();
    let initial_keys: Vec<f32> = (0..element_count).map(|idx| 1_000.0 + idx as f32).collect();
    let initial_values: Vec<f32> = (0..element_count).map(|idx| 2_000.0 + idx as f32).collect();

    layer.keys.write_typed(context, &initial_keys, 0).unwrap();
    layer.values.write_typed(context, &initial_values, 0).unwrap();

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

    let (initial_keys, initial_values) = fill_arrays(context, &mut layer);

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

    let actual_keys = layer.keys.read_typed::<f32>(context).unwrap().to_vec();
    assert_eq!(actual_keys, expected_keys, "{}: key buffer mismatch", scenario.name);

    let actual_values = layer.values.read_typed::<f32>(context).unwrap().to_vec();
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
    let (initial_keys, initial_values) = fill_arrays(context.as_ref(), &mut layer);

    let slice = layer.slice(&context, 0..2).expect("slice should exist");
    // Mutate the captured slots.

    {
        layer.keys.write_typed(context.as_ref(), &[-1.0f32, -2.0f32], 0).unwrap();
        layer.values.write_typed(context.as_ref(), &[-3.0f32, -4.0f32], 0).unwrap();
    }

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder should exist");
    layer.apply_slice(&mut encoder, &slice, None);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let keys_after = layer.keys.read_typed::<f32>(context.as_ref()).unwrap().to_vec();
    let values_after = layer.values.read_typed::<f32>(context.as_ref()).unwrap().to_vec();
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
    let (initial_keys, initial_values) = fill_arrays(context.as_ref(), &mut layer);

    let slice = layer.slice(&context, 2..4).expect("slice should exist");
    // Captured slots are expected to wrap; mutate them.
    {
        layer.keys.write_typed(context.as_ref(), &[-11.0f32, -12.0f32], 2).unwrap();
        layer.values.write_typed(context.as_ref(), &[-13.0f32, -14.0f32], 2).unwrap();
    }

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder should exist");
    layer.apply_slice(&mut encoder, &slice, None);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let keys_after = layer.keys.read_typed::<f32>(&context).unwrap().to_vec();
    let values_after = layer.values.read_typed::<f32>(&context).unwrap().to_vec();
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

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder should exist");
    layer.apply_slice(&mut encoder, &slice, None);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    if let KVCacheLayerState::Full {
        prefix_len,
    } = &layer.state
    {
        assert_eq!(*prefix_len, 3, "full slice restores prefix_len");
    } else {
        panic!("expected full state");
    }
}
