use super::{AudioError, AudioTokenGrid};

#[test]
fn token_grid_get_reads_expected_slots() {
    let grid = AudioTokenGrid::new(
        vec![
            0, 1, 2, 3, // b0 f0, f1
            4, 5, 6, 7, // b1 f0, f1
        ]
        .into_boxed_slice(),
        2,
        2,
        2,
        vec![2, 2].into_boxed_slice(),
    )
    .expect("valid grid");

    assert_eq!(grid.get(0, 0, 0), 0);
    assert_eq!(grid.get(0, 1, 1), 3);
    assert_eq!(grid.get(1, 0, 0), 4);
    assert_eq!(grid.get(1, 1, 1), 7);
}

#[test]
fn invalid_grid_shape_is_rejected() {
    let error = AudioTokenGrid::new(vec![0, 1, 2].into_boxed_slice(), 1, 2, 2, vec![2].into_boxed_slice())
        .expect_err("shape mismatch should fail");

    assert!(matches!(
        error,
        AudioError::InvalidTokenShape {
            expected_tokens: 4,
            actual_tokens: 3
        }
    ));
}
