use std::collections::{BTreeMap, BTreeSet};

use rhai::{Dynamic, Engine, Module, Scope};

#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::common::mangling::unqualify_variant;

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn unqualify_variant(value: &str) -> &str {
    value.rsplit("::").next().unwrap_or(value)
}

pub struct Evaluator {
    engine: Engine,
}

impl Evaluator {
    pub fn new<'a>(variant_values: impl IntoIterator<Item = &'a str>) -> Self {
        let mut engine = Engine::new();
        let mut namespaces: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for value in variant_values {
            if let Some((namespace, name)) = value.rsplit_once("::") {
                namespaces.entry(namespace).or_default().insert(name);
            }
        }
        for (namespace, names) in namespaces {
            let mut module = Module::new();
            for name in names {
                module.set_var(name, name.to_string());
            }
            engine.register_static_module(namespace, module.into());
        }
        Self {
            engine,
        }
    }

    pub fn satisfied<N: AsRef<str>, V: AsRef<str>>(
        &self,
        bindings: &[(N, V)],
        constraints: &[impl AsRef<str>],
    ) -> bool {
        if constraints.is_empty() {
            return true;
        }
        let mut scope = Scope::with_capacity(bindings.len());
        for (name, val) in bindings {
            let val = unqualify_variant(val.as_ref());
            scope.push(
                name.as_ref().to_owned(),
                self.engine.eval_expression::<Dynamic>(val).unwrap_or_else(|_| val.to_owned().into()),
            );
        }
        constraints.iter().all(|c| {
            self.engine
                .eval_expression_with_scope::<bool>(&mut scope, c.as_ref())
                .unwrap_or_else(|e| panic!("constraint `{}` failed to evaluate: {e}", c.as_ref()))
        })
    }
}
