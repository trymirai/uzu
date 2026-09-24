use uzu_engine_macros::uzu_test;

use super::{DFlashTfmTreeConstructionMethod, DFlashTfmTreeShape};

fn weaver_shape(construction_method: &str) -> DFlashTfmTreeShape {
    serde_json::from_str(&format!(
        r#"{{"tree_budget":16,"max_tree_depth":16,"dflash_depth_override":null,"construction_method":{construction_method}}}"#
    ))
    .unwrap()
}

/// Shapes written before prune noise existed carry no `prune_sigma` and must keep pruning on the model logprobs.
#[uzu_test]
fn weaver_shape_prune_sigma_is_optional() {
    let without = weaver_shape(r#"{"type":"Weaver","rounds":16,"expand_per_round":4,"expand_width":4}"#);
    assert_eq!(
        without.construction_method,
        DFlashTfmTreeConstructionMethod::Weaver {
            rounds: 16,
            expand_per_round: 4,
            expand_width: 4,
            prune_sigma: None,
        }
    );
    let with = weaver_shape(r#"{"type":"Weaver","rounds":16,"expand_per_round":4,"expand_width":4,"prune_sigma":1.5}"#);
    assert_eq!(
        with.construction_method,
        DFlashTfmTreeConstructionMethod::Weaver {
            rounds: 16,
            expand_per_round: 4,
            expand_width: 4,
            prune_sigma: Some(1.5),
        }
    );
}
