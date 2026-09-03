use std::collections::{BTreeMap, BTreeSet};

#[cfg(all(feature = "metal", target_os = "macos"))]
use rhai::EvalAltResult;
use rhai::{AST, Dynamic, Engine, Module, Scope};

struct CompiledConstraint {
    source: Box<str>,
    ast: AST,
}

pub struct Constraints {
    engine: Engine,
    constraints: Box<[CompiledConstraint]>,
}

impl Constraints {
    pub fn new<'a>(
        variant_values: impl IntoIterator<Item = &'a str>,
        constraints: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Self {
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
        let constraints = constraints
            .into_iter()
            .map(|constraint| {
                let source: Box<str> = constraint.as_ref().into();
                let ast = engine
                    .compile_expression(&source)
                    .unwrap_or_else(|error| panic!("constraint `{source}` failed to compile: {error}"));
                CompiledConstraint {
                    source,
                    ast,
                }
            })
            .collect();
        Self {
            engine,
            constraints,
        }
    }

    fn scope(
        &self,
        bindings: impl IntoIterator<Item = (impl AsRef<str>, impl AsRef<str>)>,
    ) -> Scope<'static> {
        let bindings = bindings.into_iter();
        let mut scope = Scope::with_capacity(bindings.size_hint().0);
        for (name, val) in bindings {
            let name = name.as_ref();
            let val = val.as_ref();
            let val = val.rsplit_once("::").map_or(val, |(_, val)| val);
            scope.push(
                name.to_owned(),
                self.engine.eval_expression::<Dynamic>(val).unwrap_or_else(|_| val.to_owned().into()),
            );
        }
        scope
    }

    /// Test partial bindings, treating a missing variable as a constraint that may still become true.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn could_satisfy(
        &self,
        bindings: impl IntoIterator<Item = (impl AsRef<str>, impl AsRef<str>)>,
    ) -> bool {
        if self.constraints.is_empty() {
            return true;
        }

        let mut scope = self.scope(bindings);
        self.constraints.iter().all(|constraint| {
            match self.engine.eval_ast_with_scope::<bool>(&mut scope, &constraint.ast) {
                Ok(satisfied) => satisfied,
                Err(error) if matches!(error.as_ref(), EvalAltResult::ErrorVariableNotFound(..)) => true,
                Err(error) => panic!("constraint `{}` failed to evaluate: {error}", constraint.source),
            }
        })
    }

    /// Test complete bindings, requiring every constraint to evaluate successfully to `true`.
    pub fn satisfied(
        &self,
        bindings: impl IntoIterator<Item = (impl AsRef<str>, impl AsRef<str>)>,
    ) -> bool {
        if self.constraints.is_empty() {
            return true;
        }

        let mut scope = self.scope(bindings);
        self.constraints.iter().all(|constraint| {
            self.engine
                .eval_ast_with_scope::<bool>(&mut scope, &constraint.ast)
                .unwrap_or_else(|error| panic!("constraint `{}` failed to evaluate: {error}", constraint.source))
        })
    }
}
