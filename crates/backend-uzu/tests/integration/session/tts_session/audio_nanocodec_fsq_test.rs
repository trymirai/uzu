use crate::common::audio::fsq_reference::{fsq_decode_reference, fsq_encode_reference};

#[test]
fn fsq_decode_reference_masks_values_beyond_lengths() {
    let output = fsq_decode_reference(&[0, 7, 11, 3, 5, 9], &[2], 1, 2, 3, 2, &[8, 5]).expect("decode");

    assert_eq!(output.len(), 12);
    let masked_t2 = [output[2], output[5], output[8], output[11]];
    assert_eq!(masked_t2, [0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn fsq_encode_reference_masks_values_beyond_lengths() {
    let tokens = fsq_encode_reference(
        &[-0.9, -0.1, 0.2, 0.7, 0.8, 0.3, -0.4, -0.7, -0.2, 0.0, 0.4, 0.9, 0.5, -0.5, 0.1, -0.1],
        &[3],
        1,
        2,
        4,
        2,
        &[8, 6],
        &[1, 8],
        1e-3,
    )
    .expect("encode");

    assert_eq!(tokens.len(), 8);
    assert_eq!(tokens[3], 0);
    assert_eq!(tokens[7], 0);
}
