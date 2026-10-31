use std::collections::HashMap;

use uzu_engine_macros::uzu_test;

use super::{DFlashTfmTreeConstructionMethod, DFlashTfmTreeShape, select_shape};
use crate::backends::common::DeviceCapabilities;

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

/// A listed device's shape wins, then the tensor-core shape where the device takes it, then the listed default; an
/// unlisted device speculates without a default only when untuned, with the smallest budget, ties broken by name.
#[uzu_test]
fn select_shape_prefers_device_tensor_cores_default_then_smallest() {
    let shape = |tree_budget, max_tree_depth| DFlashTfmTreeShape {
        tree_budget,
        max_tree_depth,
        dflash_depth_override: None,
        construction_method: DFlashTfmTreeConstructionMethod::Argmax,
    };
    let listed =
        [("GPU", shape(4, 4)), ("default", shape(8, 8)), ("b", shape(2, 3)), ("a", shape(2, 5)), ("c", shape(3, 3))];
    let tensor_cores = DeviceCapabilities::TENSOR_CORE_SPECULATION;
    let untuned = DeviceCapabilities::UNTUNED_SPECULATION;
    for (names, capabilities, tensor_core_shape, expected) in [
        (&["GPU", "default", "a"][..], tensor_cores | untuned, Some(shape(16, 12)), Some(shape(4, 4))),
        (&["default", "a"], tensor_cores | untuned, Some(shape(16, 12)), Some(shape(16, 12))),
        (&["default", "a"], tensor_cores, None, Some(shape(8, 8))),
        (&["default", "a"], untuned, Some(shape(16, 12)), Some(shape(8, 8))),
        (&["a", "b", "c"], DeviceCapabilities::empty(), Some(shape(16, 12)), None),
        (&["c", "b", "a"], untuned, None, Some(shape(2, 5))),
    ] {
        let shapes = listed
            .iter()
            .filter(|(name, _)| names.contains(name))
            .map(|(name, shape)| (name.to_string(), shape.clone()))
            .collect::<HashMap<_, _>>();
        assert_eq!(select_shape(Some("GPU"), capabilities, tensor_core_shape, shapes), expected, "{names:?}");
    }
}
