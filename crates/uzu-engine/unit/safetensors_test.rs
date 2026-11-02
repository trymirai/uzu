use uzu_engine_macros::uzu_test;

use super::shape_as_read;

/// Qwen3.5-35B-A3B stores its shared expert as a mixture of one expert, next to the 256 routed experts.
#[uzu_test]
fn unit_mixture_reads_the_one_expert() {
    let cases: [(&str, &[u32], bool, Option<&[u32]>); 7] = [
        ("shared expert up weights", &[1, 1024, 1024], true, Some(&[1024, 1024])),
        ("shared expert up scales", &[1, 32, 1024], true, Some(&[32, 1024])),
        ("shared expert input signs", &[1, 2048], true, Some(&[2048])),
        ("routed expert up weights", &[256, 1024, 1024], true, None),
        ("matrix without the expert dimension", &[1024, 1024], true, None),
        ("shared expert up weights outside a unit mixture", &[1, 1024, 1024], false, Some(&[1, 1024, 1024])),
        ("gate input signs", &[2048], false, Some(&[2048])),
    ];

    for (tensor, stored_shape, unit_mixture, expected) in cases {
        assert_eq!(shape_as_read(stored_shape, unit_mixture), expected, "{tensor}");
    }
}
