use std::sync::Arc;

use shoji::{
    traits::{
        Backend,
        backend::{chat_message::Instance as ChatMessageInstance, chat_token::Instance as ChatTokenInstance},
    },
    types::{
        model::{Model, ModelSpecialization},
        session::chat::ChatConfig,
    },
};

use crate::chat::{ChatSessionError, message, token};

#[derive(Clone)]
pub enum ChatInstanceKind {
    Token(Arc<dyn ChatTokenInstance>),
    Message(Arc<dyn ChatMessageInstance>),
}

/// A loaded chat backend instance (model weights, tokenizer, configuration).
/// Cloning is cheap and shares the underlying instance, so multiple [`ChatSession`]s
/// can be created from one [`ChatInstance`] without loading the model again.
#[bindings::export(Class)]
#[derive(Clone)]
pub struct ChatInstance {
    kind: ChatInstanceKind,
    model: Model,
    reference: String,
}

impl ChatInstance {
    pub async fn new(
        backend: Arc<dyn Backend>,
        config: ChatConfig,
        model: Model,
        path: Option<String>,
    ) -> Result<Self, ChatSessionError> {
        if !model.specializations.contains(&ModelSpecialization::Chat {}) {
            return Err(ChatSessionError::UnsupportedModel {});
        }
        let reference = path.unwrap_or_else(|| model.identifier.clone());

        let kind = tokio::spawn({
            let reference = reference.clone();
            async move {
                if let Some(token_backend) = backend.as_chat_via_token_capable() {
                    token::Session::create_instance(token_backend, config, reference).await.map(ChatInstanceKind::Token)
                } else if let Some(message_backend) = backend.as_chat_via_message_capable() {
                    message::Session::create_instance(message_backend, config, reference)
                        .await
                        .map(ChatInstanceKind::Message)
                } else {
                    Err(ChatSessionError::UnsupportedModel {})
                }
            }
        })
        .await
        .map_err(|error| ChatSessionError::Backend {
            message: error.to_string(),
        })??;

        Ok(Self {
            kind,
            model,
            reference,
        })
    }

    pub fn kind(&self) -> ChatInstanceKind {
        self.kind.clone()
    }

    pub fn reference(&self) -> String {
        self.reference.clone()
    }
}

#[bindings::export(Implementation)]
impl ChatInstance {
    #[bindings::export(Method(Getter))]
    pub fn model(&self) -> Model {
        self.model.clone()
    }
}
