use rhai::{Dynamic, Engine, Scope};

/// Returns true if all constraints pass for the given name/value bindings.
/// Panics on eval failure (programming error in kernel definition).
pub fn satisfied<N: AsRef<str>, V: AsRef<str>>(
    engine: &Engine,
    bindings: &[(N, V)],
    constraints: &[impl AsRef<str>],
) -> bool {
    if constraints.is_empty() {
        return true;
    }
    let mut scope = Scope::with_capacity(bindings.len());
    for (name, val) in bindings {
        let val = val.as_ref();
        scope.push(
            name.as_ref().to_owned(),
            engine.eval_expression::<Dynamic>(val).unwrap_or_else(|_| val.to_owned().into()),
        );
    }
    constraints.iter().all(|c| {
        engine
            .eval_expression_with_scope::<bool>(&mut scope, c.as_ref())
            .unwrap_or_else(|e| panic!("constraint `{}` failed to evaluate: {e}", c.as_ref()))
    })
}
