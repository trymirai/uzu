use uzu_engine_macros::uzu_test;

use super::{DFlashTfmDraftSampling, DFlashTfmTreeConstructionMethod, DFlashTfmTreeShape};
use crate::encodable_block::{sampling::SamplingMethod, weaver::WeaverDraftSampling};

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
        draft_sampling: None,
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
                draft_sampling: None,
            },
        };
        let json = serde_json::to_string(&shape).unwrap();
        assert_eq!(serde_json::from_str::<DFlashTfmTreeShape>(&json).unwrap(), shape, "{json}");
    }
}

/// `draft_sampling` is opt-in: shapes without it keep today's tree, and only the known value parses.
#[uzu_test]
fn weaver_shape_draft_sampling_is_optional() {
    let method = |draft_sampling: &str| {
        format!(r#"{{"type":"Weaver","rounds":16,"expand_per_round":4,"expand_width":4{draft_sampling}}}"#)
    };
    let draft_sampling = |shape: DFlashTfmTreeShape| match shape.construction_method {
        DFlashTfmTreeConstructionMethod::Weaver {
            draft_sampling,
            ..
        } => draft_sampling,
        DFlashTfmTreeConstructionMethod::Argmax => unreachable!(),
    };
    assert_eq!(draft_sampling(weaver_shape(&method(""))), None);
    assert_eq!(draft_sampling(weaver_shape(&method(r#","draft_sampling":null"#))), None);
    assert_eq!(
        draft_sampling(weaver_shape(&method(r#","draft_sampling":"Target""#))),
        Some(DFlashTfmDraftSampling::Target)
    );
    let unknown = format!(
        r#"{{"tree_budget":16,"max_tree_depth":16,"dflash_depth_override":null,"construction_method":{}}}"#,
        method(r#","draft_sampling":"Verifier""#)
    );
    assert!(serde_json::from_str::<DFlashTfmTreeShape>(&unknown).is_err());
}

/// The tree follows the request's sampling only when it changes something and keeps the most likely candidate:
/// greedy, a unit temperature without filters and values the target sampler itself cannot handle turn it off.
#[uzu_test]
fn draft_sampling_follows_the_request() {
    let stochastic = |temperature, top_k, top_p, min_p| SamplingMethod::Stochastic {
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty: None,
        suffix_repetition_length: None,
    };
    assert_eq!(WeaverDraftSampling::from_sampling_method(&SamplingMethod::Greedy), None);
    for off in [
        stochastic(None, None, None, None),
        stochastic(Some(1.0), None, None, None),
        stochastic(Some(0.0), Some(20), None, None),
        stochastic(Some(f32::INFINITY), Some(20), None, None),
        stochastic(Some(f32::NAN), None, None, None),
        stochastic(None, Some(0), None, None),
        stochastic(None, None, Some(0.0), None),
        stochastic(None, None, Some(f32::NAN), None),
        stochastic(None, None, None, Some(1.5)),
    ] {
        assert_eq!(WeaverDraftSampling::from_sampling_method(&off), None, "{off:?}");
    }
    assert_eq!(
        WeaverDraftSampling::from_sampling_method(&stochastic(Some(0.7), None, None, None)),
        Some(WeaverDraftSampling {
            temperature: Some(0.7),
            top_k: None,
            top_p: None,
            min_p: None,
        })
    );
    let full =
        WeaverDraftSampling::from_sampling_method(&stochastic(Some(0.7), Some(20), Some(0.95), Some(0.05))).unwrap();
    assert_eq!(
        full,
        WeaverDraftSampling {
            temperature: Some(0.7),
            top_k: Some(20),
            top_p: Some(0.95),
            min_p: Some(0.05),
        }
    );
    let params = full.params();
    assert_eq!((params.recip_temperature, params.top_k, params.top_p, params.min_p), (1.0 / 0.7, 20, 0.95, 0.05));
    let unset = WeaverDraftSampling::from_sampling_method(&stochastic(None, Some(20), None, None)).unwrap().params();
    assert_eq!((unset.recip_temperature, unset.top_p, unset.min_p), (1.0, f32::MAX, 0.0));
}
