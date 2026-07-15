use std::collections::HashMap;

use shoji::types::basic::{ToolDescription, ToolFunction, ToolNamespace};

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
        self.functions.insert(function.name().to_string(), function);
    }

    pub fn get_function(
        &self,
        name: &str,
    ) -> Option<&ToolFunctionDefinition> {
        self.functions.get(name)
    }

    pub fn get_namespaces(&self) -> Vec<ToolNamespace> {
        let mut namespaces = Vec::new();

        let tools_functions = self
            .functions
            .values()
            .map(|func_def| ToolDescription::Function {
                tool_function: ToolFunction {
                    name: func_def.name().to_string(),
                    description: func_def.description().to_string(),
                    parameters: func_def.parameters().clone(),
                    return_definition: func_def.return_definition().clone(),
                },
            })
            .collect::<Vec<_>>();
        if !tools_functions.is_empty() {
            namespaces.push(ToolNamespace {
                name: "functions".to_string(),
                description: None,
                tools: tools_functions,
            });
        }

        namespaces
    }
}
