use openai_harmony::chat::{
    ToolDescription as ExternalToolDescription, ToolNamespaceConfig as ExternalToolNamespaceConfig,
};
use shoji::types::{ToolDescription, ToolFunction, ToolNamespace, Value};

use crate::chat::harmony::bridging::{Error, FromHarmony, ToHarmony};

impl ToHarmony for ToolNamespace {
    type Output = ExternalToolNamespaceConfig;

    fn to_harmony(self) -> Result<Self::Output, Error> {
        let tools = self.tools.into_iter().map(ToolDescription::to_harmony).collect::<Result<Vec<_>, _>>()?;

        Ok(ExternalToolNamespaceConfig::new(self.name, self.description, tools))
    }
}

impl FromHarmony for ToolNamespace {
    type Input = ExternalToolNamespaceConfig;

    fn from_harmony(input: Self::Input) -> Self {
        let tools = input.tools.into_iter().map(ToolDescription::from_harmony).collect();

        ToolNamespace {
            name: input.name,
            description: input.description,
            tools,
        }
    }
}

impl ToHarmony for ToolDescription {
    type Output = ExternalToolDescription;

    fn to_harmony(self) -> Result<Self::Output, Error> {
        match self {
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

impl FromHarmony for ToolDescription {
    type Input = ExternalToolDescription;

    fn from_harmony(input: Self::Input) -> Self {
        ToolDescription::Function {
            function: ToolFunction {
                name: input.name,
                description: input.description,
                parameters: input.parameters.map(Value::from),
                r#return: None,
            },
        }
    }
}
