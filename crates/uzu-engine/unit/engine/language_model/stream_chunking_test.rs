use uzu_engine_macros::uzu_test;

use super::{MAX_PREFILL_CHUNK_SIZE, pipelined_prefill_chunk_lengths, prefill_chunk_lengths};

fn chunk_lengths(prompt_length: usize) -> Vec<usize> {
    pipelined_prefill_chunk_lengths(prompt_length)
}

#[uzu_test]
fn pipelined_chunks_split_short_and_medium_prompts() {
    assert_eq!(chunk_lengths(1024), [768, 256]);
    assert_eq!(chunk_lengths(1536), [1024, 512]);
    assert_eq!(chunk_lengths(2048), [1024, 1024]);
    assert_eq!(chunk_lengths(768), [512, 256]);
    assert_eq!(chunk_lengths(600), [300, 300]);
    assert_eq!(chunk_lengths(4096), [1024; 4]);
}

#[uzu_test]
fn pipelined_chunks_leave_tiny_prompts_whole() {
    for length in [1, 128, 256, 512] {
        assert_eq!(chunk_lengths(length), [length]);
    }
}

#[uzu_test]
fn pipelined_chunks_cover_the_prompt_without_tiny_tails() {
    for length in 1..=9000 {
        let chunks = chunk_lengths(length);
        assert_eq!(chunks.iter().sum::<usize>(), length);
        assert!(chunks.iter().all(|chunk| *chunk <= MAX_PREFILL_CHUNK_SIZE), "{length}: {chunks:?}");
        if length > 512 {
            assert!(chunks.len() > 1, "{length}: {chunks:?}");
            assert!(*chunks.last().unwrap() >= 128, "{length}: {chunks:?}");
        }
    }
}

#[uzu_test]
fn unpipelined_chunks_are_full_chunks() {
    assert_eq!(prefill_chunk_lengths(1024, false), [1024]);
    assert_eq!(prefill_chunk_lengths(3000, false), [1024, 1024, 952]);
    assert_eq!(prefill_chunk_lengths(1, false), [1]);
}
