use std::collections::HashMap;

use crate::tool::func_def::ToolFunctionDefinition;

#[derive(Clone)]
pub struct ToolRegistry {
    functions: HashMap<String, ToolFunctionDefinition>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    pub fn add_function(
        &mut self,
        function: ToolFunctionDefinition,
    ) {
        self.functions.insert(function.name(), function);
    }
}
