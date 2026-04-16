use openai_harmony::chat::{
    ToolDescription as ExternalToolDescription, ToolNamespaceConfig as ExternalToolNamespaceConfig,
};

use crate::chat::{
    encoding::harmony::bridging::Error,
    types::{ToolDescription, ToolFunction, ToolNamespace, Value},
};

impl TryFrom<ToolNamespace> for ExternalToolNamespaceConfig {
    type Error = Error;

    fn try_from(namespace: ToolNamespace) -> Result<ExternalToolNamespaceConfig, Error> {
        let tools =
            namespace.tools.into_iter().map(ExternalToolDescription::try_from).collect::<Result<Vec<_>, _>>()?;

        Ok(ExternalToolNamespaceConfig::new(namespace.name, namespace.description, tools))
    }
}

impl TryFrom<ToolDescription> for ExternalToolDescription {
    type Error = Error;

    fn try_from(tool: ToolDescription) -> Result<ExternalToolDescription, Error> {
        match tool {
            ToolDescription::Function {
                function,
            } => {
                let parameters = function.parameters.map(serde_json::Value::try_from).transpose().map_err(|error| {
                    Error::SerializationFailed {
                        message: error.to_string(),
                    }
                })?;

                Ok(ExternalToolDescription::new(function.name, function.description, parameters))
            },
        }
    }
}

impl From<ExternalToolNamespaceConfig> for ToolNamespace {
    fn from(namespace: ExternalToolNamespaceConfig) -> ToolNamespace {
        let tools = namespace.tools.into_iter().map(ToolDescription::from).collect();

        ToolNamespace {
            name: namespace.name,
            description: namespace.description,
            tools,
        }
    }
}

impl From<ExternalToolDescription> for ToolDescription {
    fn from(tool: ExternalToolDescription) -> ToolDescription {
        ToolDescription::Function {
            function: ToolFunction {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters.map(Value::from),
                r#return: None,
            },
        }
    }
}
