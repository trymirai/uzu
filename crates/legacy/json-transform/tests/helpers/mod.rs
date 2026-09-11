#![allow(dead_code)]

use indexmap::IndexMap;
use json_transform::{
    TransformError, TransformSchema,
    execution::{GetTarget, Operation, PathSegment, Pipeline},
};
use tracing_subscriber::{EnvFilter, Registry, prelude::*};
use tracing_tree::HierarchicalLayer;

pub fn init_tracing_for_tests() {
    let filter = EnvFilter::from_default_env().add_directive("json_transform".parse().unwrap());

    let layer = HierarchicalLayer::new(2).with_bracketed_fields(true).with_targets(true);

    let _ = Registry::default().with(filter).with(layer).try_init();
}

pub fn get_key(key: &str) -> Operation {
    Operation::Get {
        target: GetTarget::Key {
            key: key.to_string(),
        },
    }
}

pub fn get_path(segments: Vec<PathSegment>) -> Operation {
    Operation::Get {
        target: GetTarget::Path {
            path: segments,
        },
    }
}

pub fn schema_with_root(root: Pipeline) -> TransformSchema {
    TransformSchema {
        pipelines: IndexMap::from([("root".to_string(), root)]),
    }
}

pub fn execute_root(
    root: Pipeline,
    input: serde_json::Value,
) -> Result<serde_json::Value, TransformError> {
    init_tracing_for_tests();
    schema_with_root(root).execute("root", input)
}
