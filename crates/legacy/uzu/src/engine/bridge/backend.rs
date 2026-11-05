use std::pin::Pin;

use shoji::{
    traits::{
        Backend as LlmBackend,
        backend::{
            Error as BackendError,
            chat_token::{Backend as ChatTokenBackend, Instance as ChatTokenInstance},
            classification::{Backend as ClassificationBackend, Config, Instance as ClassificationInstance},
            decision::{Config as DecisionConfig, DecisionBackend, DecisionInstance},
        },
    },
    types::session::chat::ChatConfig,
};
use uzu_engine::{
    TOOLCHAIN_VERSION,
    backends::{Backend, BackendSelection, select_backend},
};

use crate::engine::{
    MemoryCheck,
    bridge::{
        chat_token_backend::UzuChatTokenBackendInstance, classification_backend::UzuClassificationBackendInstance,
        decision_backend::UzuDecisionBackendInstance,
    },
};

struct ChatTokenInstanceSelection {
    reference: String,
    config: ChatConfig,
    memory_check: Option<MemoryCheck>,
}

impl BackendSelection for ChatTokenInstanceSelection {
    type Output = Box<dyn ChatTokenInstance>;
    type Error = BackendError;

    async fn select<B: Backend>(self) -> Result<Self::Output, Self::Error> {
        UzuChatTokenBackendInstance::<B>::new(self.reference, self.config, self.memory_check)
            .await
            .map(|instance| Box::new(instance) as Box<dyn ChatTokenInstance>)
    }
}

struct ClassificationInstanceSelection {
    reference: String,
}

impl BackendSelection for ClassificationInstanceSelection {
    type Output = Box<dyn ClassificationInstance>;
    type Error = BackendError;

    async fn select<B: Backend>(self) -> Result<Self::Output, Self::Error> {
        UzuClassificationBackendInstance::<B>::new(self.reference)
            .await
            .map(|instance| Box::new(instance) as Box<dyn ClassificationInstance>)
    }
}

struct DecisionInstanceSelection {
    reference: String,
}

impl BackendSelection for DecisionInstanceSelection {
    type Output = Box<dyn DecisionInstance>;
    type Error = BackendError;

    async fn select<B: Backend>(self) -> Result<Self::Output, Self::Error> {
        UzuDecisionBackendInstance::<B>::new(self.reference)
            .await
            .map(|instance| Box::new(instance) as Box<dyn DecisionInstance>)
    }
}

/// The Uzu backend; for a chat, with the memory check its model must pass before its weights load.
pub struct UzuLlmBackend {
    memory_check: Option<MemoryCheck>,
}

impl UzuLlmBackend {
    pub fn new() -> Self {
        Self {
            memory_check: None,
        }
    }

    pub fn with_memory_check(memory_check: MemoryCheck) -> Self {
        Self {
            memory_check: Some(memory_check),
        }
    }
}

impl LlmBackend for UzuLlmBackend {
    fn identifier(&self) -> String {
        "uzu".to_string()
    }

    fn version(&self) -> String {
        TOOLCHAIN_VERSION.to_string()
    }

    fn as_chat_via_token_capable(&self) -> Option<&dyn ChatTokenBackend> {
        Some(self)
    }

    fn as_classification_capable(&self) -> Option<&dyn ClassificationBackend> {
        Some(self)
    }

    fn as_decision_capable(&self) -> Option<&dyn DecisionBackend> {
        Some(self)
    }
}

impl ChatTokenBackend for UzuLlmBackend {
    fn instance<'a>(
        &'a self,
        reference: String,
        config: ChatConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn ChatTokenInstance>, BackendError>> + Send + 'a>> {
        Box::pin(async move {
            let instance = select_backend(
                ChatTokenInstanceSelection {
                    reference,
                    config,
                    memory_check: self.memory_check.clone(),
                },
                BackendError::from("Unable to open any backend"),
            )
            .await?;
            Ok(instance)
        })
    }
}

impl ClassificationBackend for UzuLlmBackend {
    fn instance(
        &self,
        reference: String,
        _config: Config,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn ClassificationInstance>, BackendError>> + Send + '_>> {
        Box::pin(async move {
            let instance = select_backend(
                ClassificationInstanceSelection {
                    reference,
                },
                BackendError::from("Unable to open any backend"),
            )
            .await?;
            Ok(instance)
        })
    }
}

impl DecisionBackend for UzuLlmBackend {
    fn instance(
        &self,
        reference: String,
        _config: DecisionConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn DecisionInstance>, BackendError>> + Send + '_>> {
        Box::pin(async move {
            let instance = select_backend(
                DecisionInstanceSelection {
                    reference,
                },
                BackendError::from("Unable to open any backend"),
            )
            .await?;
            Ok(instance)
        })
    }
}
