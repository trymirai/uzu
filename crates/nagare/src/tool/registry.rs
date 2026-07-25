use indexmap::IndexMap;
use shoji::types::basic::{ToolDescription, ToolFunction, ToolNamespace};

use crate::tool::func_def::ToolFunctionDefinition;

#[derive(Clone)]
pub struct ToolRegistry {
    functions: IndexMap<String, ToolFunctionDefinition>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            functions: IndexMap::new(),
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
                    parameters: func_def.parameters().clone().or_else(|| {
                        Some(
                            serde_json::json!({
                                "type": "object",
                                "properties": {},
                                "required": []
                            })
                            .into(),
                        )
                    }),
                    // No chat format documents a "return" schema in tool declarations
                    // (harmony drops it too). Small models mimic any extra key they see:
                    // Llama 3.2 1B answers with {"return": ...} echoes instead of text.
                    return_definition: None,
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
