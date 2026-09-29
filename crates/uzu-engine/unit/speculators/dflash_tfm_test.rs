use uzu_engine_macros::uzu_test;

use super::{DFlashTfmTreeConstructionMethod, DFlashTfmTreeShape};

fn weaver_shape(construction_method: &str) -> DFlashTfmTreeShape {
    serde_json::from_str(&format!(
        r#"{{"tree_budget":16,"max_tree_depth":16,"dflash_depth_override":null,"construction_method":{construction_method}}}"#
    ))
    .unwrap()
}

/// A shape without `prune_sigma` prunes with noise scale 1.5, `null` turns the noise off, and a number is the scale
/// itself, 0 included: parsing does not reinterpret it, the shape check rejects it.
#[uzu_test]
fn weaver_shape_prune_sigma_is_optional() {
    let weaver = |prune_sigma| DFlashTfmTreeConstructionMethod::Weaver {
        rounds: 16,
        expand_per_round: 4,
        expand_width: 4,
        prune_sigma,
    };
    let method = |prune_sigma: &str| {
        format!(r#"{{"type":"Weaver","rounds":16,"expand_per_round":4,"expand_width":4{prune_sigma}}}"#)
    };
    for (field, expected) in [
        ("", Some(1.5)),
        (r#","prune_sigma":null"#, None),
        (r#","prune_sigma":1.5"#, Some(1.5)),
        (r#","prune_sigma":2"#, Some(2.0)),
        (r#","prune_sigma":0"#, Some(0.0)),
    ] {
        assert_eq!(weaver_shape(&method(field)).construction_method, weaver(expected), "{field:?}");
    }
}

/// A shape written back to JSON reads as the same shape, with the noise on or off.
#[uzu_test]
fn weaver_shape_prune_sigma_round_trips() {
    for prune_sigma in [Some(1.5), Some(2.0), None] {
        let shape = DFlashTfmTreeShape {
            tree_budget: 16,
            max_tree_depth: 16,
            dflash_depth_override: None,
            construction_method: DFlashTfmTreeConstructionMethod::Weaver {
                rounds: 16,
                expand_per_round: 4,
                expand_width: 4,
                prune_sigma,
            },
        };
        let json = serde_json::to_string(&shape).unwrap();
        assert_eq!(serde_json::from_str::<DFlashTfmTreeShape>(&json).unwrap(), shape, "{json}");
    }
}
