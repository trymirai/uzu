use std::collections::HashMap;

use backend_uzu::prelude::{FixedTokensSpeculator, Speculator};

#[test]
fn test_fixed_token_speculator_single() {
    let speculator = FixedTokensSpeculator::new(vec![vec![0, 1, 2, 3, 4]]);

    assert_eq!(speculator.speculate(&[]), HashMap::from([(0, 1.0)]));
    assert_eq!(speculator.speculate(&[1337]), HashMap::from([(0, 1.0)]));

    assert_eq!(speculator.speculate(&[0]), HashMap::from([(1, 1.0)]));

    assert_eq!(speculator.speculate(&[0, 1]), HashMap::from([(2, 1.0)]));
    assert_eq!(speculator.speculate(&[1]), HashMap::from([(2, 1.0)]));

    assert_eq!(speculator.speculate(&[4]), HashMap::from([(0, 1.0)]));
}

#[test]
fn test_fixed_token_speculator_multi() {
    let speculator =
        FixedTokensSpeculator::new(vec![vec![0, 1, 2, 3, 4], vec![0, 1, 2, 4, 8], vec![0, 10, 20, 30, 40]]);

    assert_eq!(speculator.speculate(&[]), HashMap::from([(0, 1.0)]));
    assert_eq!(speculator.speculate(&[1337]), HashMap::from([(0, 1.0)]));

    assert_eq!(speculator.speculate(&[0]), HashMap::from([(1, 2.0 / 3.0), (10, 1.0 / 3.0)]));

    assert_eq!(speculator.speculate(&[0, 1]), HashMap::from([(0, 1.0 / 3.0), (2, 2.0 / 3.0)]));
    assert_eq!(speculator.speculate(&[1]), HashMap::from([(0, 1.0 / 3.0), (2, 2.0 / 3.0)]));

    assert_eq!(speculator.speculate(&[4]), HashMap::from([(8, 1.0 / 3.0), (0, 2.0 / 3.0)]));
}
