use backend_uzu::prelude::TokenFinder;

#[test]
fn test_find_candidate_pred_token_empty_sequence() {
    let sequence: Vec<u64> = vec![];

    let result = TokenFinder::find_candidate_pred_token(&sequence, 3);

    assert!(result.is_none());
}

#[test]
fn test_find_candidate_pred_token_sequence_too_short() {
    let sequence: Vec<u64> = vec![1, 2];

    let result = TokenFinder::find_candidate_pred_token(&sequence, 3);

    assert!(result.is_none());
}

#[test]
fn test_find_candidate_pred_token_exact_pattern_match() {
    let sequence = vec![1, 2, 3, 4, 5, 1, 2, 3];

    let result = TokenFinder::find_candidate_pred_token(&sequence, 3);

    assert_eq!(result, Some(4));
}

#[test]
fn test_find_candidate_pred_token_valid_match() {
    let mut sequence = vec![1, 2, 3, 4, 5, 1, 2, 3];

    let result1 = TokenFinder::find_candidate_pred_token(&sequence, 3);

    assert_eq!(result1, Some(4));

    sequence.push(4);

    let result2 = TokenFinder::find_candidate_pred_token(&sequence, 3);

    assert_eq!(result2, Some(5));
}

#[test]
fn test_find_candidate_pred_token_multiple_matches() {
    let mut sequence = vec![1, 2, 3, 4, 5, 1, 2, 3, 6, 7, 1, 2, 3];

    let result1 = TokenFinder::find_candidate_pred_token(&sequence, 3);

    assert_eq!(result1, Some(4));
    assert_ne!(result1, Some(6));

    sequence.push(4);

    let result2 = TokenFinder::find_candidate_pred_token(&sequence, 3);

    assert_eq!(result2, Some(5));
    assert_ne!(result2, Some(7));
}

#[test]
fn test_find_candidate_pred_token_different_ngram_sizes() {
    let mut sequence = vec![1, 2, 3, 4, 5, 1, 2, 3, 8, 9, 1, 2, 3, 4, 5];

    let result1_3 = TokenFinder::find_candidate_pred_token(&sequence, 3);
    let result1_2 = TokenFinder::find_candidate_pred_token(&sequence, 2);
    let result1_1 = TokenFinder::find_candidate_pred_token(&sequence, 1);

    assert_eq!(result1_3, Some(1));
    assert_eq!(result1_2, Some(1));
    assert_eq!(result1_1, Some(1));

    sequence.push(1);

    let result2_3 = TokenFinder::find_candidate_pred_token(&sequence, 3);
    let result2_2 = TokenFinder::find_candidate_pred_token(&sequence, 2);
    let result2_1 = TokenFinder::find_candidate_pred_token(&sequence, 1);

    assert_eq!(result2_3, Some(2));
    assert_eq!(result2_2, Some(2));
    assert_eq!(result2_1, Some(2));
}

#[test]
fn test_find_candidate_pred_token_no_match() {
    let sequence = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

    let result = TokenFinder::find_candidate_pred_token(&sequence, 3);

    assert!(result.is_none());
}

#[test]
fn test_find_candidate_pred_token_max_ngram_higher_than_input() {
    let sequence = vec![1, 2, 3, 4, 5];

    let result = TokenFinder::find_candidate_pred_token(&sequence, 8);

    assert!(result.is_none());
}
