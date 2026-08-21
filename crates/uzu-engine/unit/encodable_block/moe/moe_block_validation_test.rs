use uzu_engine_macros::uzu_test;

use super::super::{MoeBlockError, valid_model_dim, validate_expert_counts};
use crate::backends::cpu::Cpu;

#[uzu_test]
fn rejects_invalid_routed_expert_counts() {
    assert!(validate_expert_counts::<Cpu>(512, 128).is_ok());
    for routed_experts in [0, 513] {
        let error = validate_expert_counts::<Cpu>(routed_experts, 1);
        assert!(matches!(error, Err(MoeBlockError::InvalidRoutedExpertCount)));
    }
}

#[uzu_test]
fn rejects_invalid_active_expert_counts() {
    for (routed_experts, active_experts) in [(1, 0), (1, 2), (512, 129)] {
        let error = validate_expert_counts::<Cpu>(routed_experts, active_experts);
        assert!(matches!(error, Err(MoeBlockError::InvalidActiveExpertCount)));
    }
}

#[uzu_test]
fn rejects_router_dimensions_beyond_its_threadgroup_cache() {
    assert!(valid_model_dim(4096));
    for model_dim in [0, 2, 4100] {
        assert!(!valid_model_dim(model_dim));
    }
}
