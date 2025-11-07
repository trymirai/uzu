use crate::session::types::Input;

#[derive(Clone, Debug)]
pub enum ContextMode {
    None,
    Static {
        input: Input,
    },
    Dynamic,
}

impl Default for ContextMode {
    fn default() -> Self {
        ContextMode::None
    }
}
