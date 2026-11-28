use uzu_engine_macros::uzu_test;

use super::{MAX_TOKENS, MIN_TOKENS, pipelined_chunk_lengths as chunks};

#[uzu_test]
fn pipelined_chunks_are_even_bounded_and_cover_the_prompt() {
    assert_eq!(chunks(2048), [1024, 1024]);
    for length in 1..=9000 {
        let chunks = chunks(length);
        assert_eq!(chunks.iter().sum::<usize>(), length);
        assert!(chunks.iter().all(|chunk| *chunk <= MAX_TOKENS as usize), "{length}: {chunks:?}");
        assert!(length < 2 * MIN_TOKENS as usize || chunks.len() % 2 == 0, "{length}: {chunks:?}");
    }
}
