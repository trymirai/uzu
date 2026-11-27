use uzu_engine_macros::uzu_test;

use super::KVCacheState;
use crate::{
    backends::cpu::Cpu,
    encodable_block::{batch_topology::BatchTopology, cell_grid::CellGrid},
};

fn accept(
    state: &mut KVCacheState,
    indices: &[u32],
) -> Vec<(u32, u32)> {
    state.accept(indices, indices.len() as u32).into_iter().map(|copy| (copy.source, copy.destination)).collect()
}

fn accept_all(
    state: &mut KVCacheState,
    batch: BatchTopology<Cpu>,
) {
    state.accept(&(0..batch.size()).collect::<Box<[u32]>>(), batch.position_advance(batch.size()));
}

#[uzu_test]
fn ring_accepts_append_then_wrap() {
    let mut state = KVCacheState::new(0, Some(2));

    assert_eq!(accept(&mut state, &[1]), [(1, 0)]);
    assert_eq!(state.prefix_len(), 1);

    assert!(accept(&mut state, &[0]).is_empty());
    assert_eq!(state.prefix_len(), 2);

    assert_eq!(accept(&mut state, &[0]), [(2, 0)]);
    assert_eq!(state.ring_params().unwrap().ring_offset, 1);
    assert_eq!(state.required_prefix_len(99), 2);
}

#[uzu_test]
fn ring_accept_over_capacity_keeps_last() {
    let mut state = KVCacheState::new(0, Some(2));

    // Only the last `capacity` tokens survive; earlier copies are dead.
    assert_eq!(accept(&mut state, &[0, 1, 2]), [(2, 0)]);
    assert_eq!(state.prefix_len(), 2);
    assert_eq!(state.ring_params().unwrap().ring_offset, 1);

    // Same with a wrapped ring: destinations keep rotating.
    assert_eq!(accept(&mut state, &[0, 1, 2]), [(3, 0), (4, 1)]);
    assert_eq!(state.ring_params().unwrap().ring_offset, 0);
}

#[uzu_test]
fn rope_position_after_image() {
    // 7 text tokens, then a 5×3 image in one batch or in two: HF puts the image at 7 and the next token at 7 + 5
    for chunks in [vec![0..15], vec![0..6, 6..15]] {
        let mut state = KVCacheState::new(0, None);
        accept_all(&mut state, BatchTopology::flat(7));
        for cells in chunks {
            assert_eq!(state.rope_position(), 7);
            accept_all(&mut state, BatchTopology::image(CellGrid::new(5, 3, 1), cells));
        }
        assert_eq!((state.context_length(), state.rope_position()), (22, 12));
        accept_all(&mut state, BatchTopology::flat(1));
        assert_eq!(state.rope_position(), 13);
    }
}
