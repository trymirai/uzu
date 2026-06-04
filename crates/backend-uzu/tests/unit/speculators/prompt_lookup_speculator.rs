use proc_macros::uzu_test;

use crate::prelude::{PromptLookupSpeculator, Speculator};

#[uzu_test]
fn test_empty_input() {
    let speculator = PromptLookupSpeculator::new();
    let prefix: Vec<u64> = vec![];

    let proposals = speculator.speculate(&prefix);

    assert_eq!(proposals.len(), 0);
}

#[uzu_test]
fn test_no_pattern_found() {
    let speculator = PromptLookupSpeculator::new();
    let prefix = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

    let proposals = speculator.speculate(&prefix);

    assert_eq!(proposals.len(), 0);
}
