use super::{AudioError, AudioTokenGrid, AudioTokenPacking};

#[test]
fn packing_conversion_roundtrip_is_lossless() {
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
        AudioTokenPacking::FrameMajor,
    )
    .expect("valid grid");

    let converted = grid.to_packing(AudioTokenPacking::CodebookMajor);
    let restored = converted.to_packing(AudioTokenPacking::FrameMajor);

    assert_eq!(restored, grid);
}

#[test]
fn invalid_grid_shape_is_rejected() {
    let error = AudioTokenGrid::new(
        vec![0, 1, 2].into_boxed_slice(),
        1,
        2,
        2,
        vec![2].into_boxed_slice(),
        AudioTokenPacking::FrameMajor,
    )
    .expect_err("shape mismatch should fail");

    assert!(matches!(
        error,
        AudioError::InvalidTokenShape {
            expected_tokens: 4,
            actual_tokens: 3
        }
    ));
}
