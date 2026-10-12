use shader_slang::{ComponentType, Module};

pub struct ModuleWithDiagnostics {
    pub module: Module,
    pub component: ComponentType,
    pub diagnostics: Option<String>,
}
