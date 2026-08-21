use std::sync::Arc;

use shoji::types::basic::ToolFunction;

use super::func_def::{ErrorFuture, ToolDescriptor};

#[derive(Debug, Clone, thiserror::Error, uniffi::Error)]
pub enum ForeignToolError {
    #[error("Tool invocation failed: {message}")]
    Invocation {
        message: String,
    },
}

#[uniffi::export(callback_interface)]
#[async_trait::async_trait]
pub trait ForeignToolHandler: Send + Sync {
    async fn invoke_json(
        &self,
        arguments_json: String,
    ) -> Result<String, ForeignToolError>;
}

#[derive(uniffi::Object)]
pub struct ForeignTool {
    descriptor: ToolDescriptor,
}

#[uniffi::export]
impl ForeignTool {
    #[uniffi::constructor]
    pub fn new(
        definition: ToolFunction,
        handler: Box<dyn ForeignToolHandler>,
    ) -> Arc<Self> {
        let handler: Arc<dyn ForeignToolHandler> = Arc::from(handler);
        let descriptor = ToolDescriptor::new(
            definition.name,
            definition.description,
            definition.parameters,
            definition.return_definition,
            Box::new(move |arguments| {
                let handler = Arc::clone(&handler);
                Box::new(async move {
                    let json = handler
                        .invoke_json(arguments.json)
                        .await
                        .map_err(|error| -> ErrorFuture { error.to_string().into() })?;
                    let value = serde_json::from_str::<serde_json::Value>(&json)?;
                    Ok(value.into())
                })
            }),
        );
        Arc::new(Self {
            descriptor,
        })
    }
}

impl ForeignTool {
    pub(crate) fn descriptor(&self) -> ToolDescriptor {
        self.descriptor.clone()
    }
}
