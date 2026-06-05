use crate::session::types::Input;

#[derive(Clone, Debug, Default)]
pub enum ContextMode {
    #[default]
    None,
    Static {
        input: Input,
    },
    Dynamic,
}

#[cfg(test)]
#[path = "../../../tests/unit/session/parameter/context_mode.rs"]
mod tests;
