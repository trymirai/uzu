use std::collections::BTreeSet;

use proptest::{collection::vec, option, prelude::*, sample::Index};
use uzu_engine_macros::uzu_test;

use super::{CacheStep, LayerId, LayerSchedule, LoopAction, ReadoutStep, ScheduleError, StateId, StepState};

fn owned(
    layer: u32,
    state: u32,
) -> CacheStep {
    CacheStep {
        layer: LayerId(layer),
        state: StepState::Owned(StateId(state)),
        loop_action: None,
        captures_hidden: false,
    }
}

fn shared(
    layer: u32,
    state: u32,
) -> CacheStep {
    CacheStep {
        state: StepState::Shared(StateId(state)),
        ..owned(layer, state)
    }
}

fn readout(
    layer: u32,
    state: u32,
) -> ReadoutStep {
    ReadoutStep {
        layer: LayerId(layer),
        state: StateId(state),
        captures_hidden: false,
    }
}

impl CacheStep {
    fn start(self) -> Self {
        Self {
            loop_action: Some(LoopAction::Start),
            ..self
        }
    }

    fn restart(self) -> Self {
        Self {
            loop_action: Some(LoopAction::Restart),
            ..self
        }
    }

    fn hidden(self) -> Self {
        Self {
            captures_hidden: true,
            ..self
        }
    }
}

fn build(
    kv_sources: &[Option<u32>],
    layer_order: Option<&[u32]>,
    hidden_targets: Option<&[u32]>,
) -> LayerSchedule {
    LayerSchedule::new(kv_sources, layer_order, hidden_targets).unwrap()
}

#[uzu_test]
fn trailing_shared_layers_become_readout_steps() {
    let schedule = build(&[None, None, Some(0), Some(1)], None, None);

    assert_eq!(*schedule.cache_steps, [owned(0, 0), owned(1, 1)]);
    assert_eq!(*schedule.readout_steps, [readout(2, 0), readout(3, 1)]);
}

#[uzu_test]
fn shared_layer_feeding_an_owning_layer_stays_a_cache_step() {
    // Layer 1 writes no state, but layer 2 does and depends on layer 1's output.
    let schedule = build(&[None, Some(0), None], None, None);

    assert_eq!(*schedule.cache_steps, [owned(0, 0), shared(1, 0), owned(2, 1)]);
    assert!(schedule.readout_steps.is_empty());
}

#[uzu_test]
fn loop_starts_at_first_visit_and_restarts_at_jump_back() {
    let schedule = build(&[None; 4], Some(&[0, 1, 2, 1, 2, 3]), None);

    assert_eq!(
        *schedule.cache_steps,
        [owned(0, 0), owned(1, 1).start(), owned(2, 2), owned(1, 3).restart(), owned(2, 4), owned(3, 5)]
    );
    assert!(schedule.readout_steps.is_empty());
}

#[uzu_test]
fn single_layer_loop_restarts_into_itself() {
    let schedule = build(&[None; 2], Some(&[0, 0, 1]), None);

    assert_eq!(*schedule.cache_steps, [owned(0, 0).start(), owned(0, 1).restart(), owned(1, 2)]);
}

#[uzu_test]
fn repeated_passes_share_one_start() {
    let schedule = build(&[None; 3], Some(&[0, 1, 1, 1, 2]), None);

    assert_eq!(
        *schedule.cache_steps,
        [owned(0, 0), owned(1, 1).start(), owned(1, 2).restart(), owned(1, 3).restart(), owned(2, 4)]
    );
}

#[uzu_test]
fn separate_loops_each_get_a_start() {
    let schedule = build(&[None; 4], Some(&[0, 1, 0, 1, 2, 3, 2, 3]), None);

    assert_eq!(
        *schedule.cache_steps,
        [
            owned(0, 0).start(),
            owned(1, 1),
            owned(0, 2).restart(),
            owned(1, 3),
            owned(2, 4).start(),
            owned(3, 5),
            owned(2, 6).restart(),
            owned(3, 7),
        ]
    );
}

#[uzu_test]
fn hidden_targets_are_step_indices() {
    // Step 2 is the second visit to layer 0; matching on layer index would capture step 0 instead.
    let schedule = build(&[None; 2], Some(&[0, 1, 0, 1]), Some(&[2]));

    assert_eq!(*schedule.cache_steps, [owned(0, 0).start(), owned(1, 1), owned(0, 2).restart().hidden(), owned(1, 3)]);
    assert_eq!(schedule.hidden_capture_count, 1);
}

#[uzu_test]
fn hidden_target_on_readout_step() {
    let schedule = build(&[None, None, Some(0)], None, Some(&[0, 2]));

    assert_eq!(*schedule.cache_steps, [owned(0, 0).hidden(), owned(1, 1)]);
    assert_eq!(
        *schedule.readout_steps,
        [ReadoutStep {
            captures_hidden: true,
            ..readout(2, 0)
        }]
    );
    assert_eq!(schedule.hidden_capture_count, 2);
    assert!(schedule.has_readout_hiddens);
}

#[uzu_test]
fn rejects_invalid_configs() {
    use ScheduleError::*;

    let cases: &[(&str, &[Option<u32>], Option<&[u32]>, Option<&[u32]>, ScheduleError)] = &[
        ("no layers", &[], None, None, NoSteps),
        ("empty order", &[None], Some(&[]), None, NoSteps),
        ("order starts past layer 0", &[None; 2], Some(&[1, 0, 1]), None, SkippedLayerIndex(0)),
        ("order skips a layer", &[None; 3], Some(&[0, 2, 1]), None, SkippedLayerIndex(1)),
        ("order never reaches the last layer", &[None; 3], Some(&[0, 1, 0, 1]), None, SkippedLayerIndex(2)),
        (
            "order references a missing layer",
            &[None],
            Some(&[0, 1]),
            None,
            LayerIndexOutOfBounds {
                index: 1,
                len: 1,
            },
        ),
        ("loop over a KV-sharing layer", &[None, Some(0)], Some(&[0, 1, 0, 1]), None, LoopingAndSharedKVCache(1)),
        ("KV source is a later layer", &[Some(1), None], None, None, SharedNotIntoOwned(0, 1)),
        ("KV source is itself shared", &[None, Some(0), Some(1)], None, None, SharedNotIntoOwned(2, 1)),
        (
            "KV source is out of range",
            &[None, Some(5)],
            None,
            None,
            LayerIndexOutOfBounds {
                index: 5,
                len: 2,
            },
        ),
        ("hidden targets unsorted", &[None; 3], None, Some(&[2, 1]), HiddenIndicesOutOfOrder(1)),
        ("hidden targets duplicated", &[None; 3], None, Some(&[1, 1]), HiddenIndicesOutOfOrder(1)),
        ("hidden target past the last step", &[None; 2], None, Some(&[2]), HiddenIndicesOutOfOrder(2)),
    ];

    for (name, kv_sources, layer_order, hidden_targets, expected) in cases {
        assert_eq!(LayerSchedule::new(kv_sources, *layer_order, *hidden_targets).unwrap_err(), *expected, "{name}");
    }
}

/// A valid looped order: each step visits either the next unvisited layer or an already visited one, and every
/// layer is visited.
fn layer_order() -> impl Strategy<Value = (usize, Vec<u32>)> {
    (1..6usize, vec(option::of(any::<Index>()), 0..16)).prop_map(|(layer_count, moves)| {
        let mut order = vec![0];
        let mut max_visited = 0;
        for revisit in moves {
            match revisit {
                Some(index) => order.push(index.index(max_visited as usize + 1) as u32),
                None if max_visited + 1 < layer_count as u32 => {
                    max_visited += 1;
                    order.push(max_visited);
                },
                None => {},
            }
        }
        order.extend(max_visited + 1..layer_count as u32);
        (layer_count, order)
    })
}

#[uzu_test]
fn looped_schedule_follows_order() {
    proptest!(|((layer_count, order) in layer_order(), hidden_mask in vec(any::<bool>(), 32))| {
        let hidden_targets = (0..order.len() as u32).filter(|&step| hidden_mask[step as usize]).collect::<Vec<_>>();
        let schedule = LayerSchedule::new(&vec![None; layer_count], Some(&order), Some(&hidden_targets)).unwrap();

        let restart_targets = order.windows(2).filter(|pair| pair[1] <= pair[0]).map(|pair| pair[1]).collect::<BTreeSet<_>>();
        let expected = order.iter().enumerate().map(|(step, &layer)| CacheStep {
            layer: LayerId(layer),
            state: StepState::Owned(StateId(step as u32)),
            loop_action: if step > 0 && layer <= order[step - 1] {
                Some(LoopAction::Restart)
            } else if restart_targets.contains(&layer) && !order[..step].contains(&layer) {
                Some(LoopAction::Start)
            } else {
                None
            },
            captures_hidden: hidden_mask[step],
        }).collect::<Vec<_>>();

        prop_assert_eq!(&*schedule.cache_steps, &*expected);
        prop_assert!(schedule.readout_steps.is_empty());
        prop_assert_eq!(schedule.hidden_capture_count as usize, hidden_targets.len());
    });
}
