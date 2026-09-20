use std::{future::Future, path::PathBuf, pin::Pin};

use shoji::{
    traits::{
        Backend as BackendTrait,
        backend::{Error as BackendError, chat_message},
    },
    types::session::chat::ChatConfig,
};

use crate::{
    config::{BACKEND_IDENTIFIER, Config, NEEDLE3_ENGINE_VERSION},
    engine::EngineHandle,
    error::Error,
    ffi::Lib,
    instance::Instance,
};

pub struct Backend {
    handle: EngineHandle,
}

impl Backend {
    pub fn try_new(config: Config) -> Result<Self, Error> {
        let lib = Lib::open(&config.lib_path)?;
        Ok(Self {
            handle: EngineHandle::new(lib),
        })
    }

    pub fn from_handle(handle: EngineHandle) -> Self {
        Self {
            handle,
        }
    }
}

impl BackendTrait for Backend {
    fn identifier(&self) -> String {
        BACKEND_IDENTIFIER.to_string()
    }

    fn version(&self) -> String {
        NEEDLE3_ENGINE_VERSION.to_string()
    }

    fn as_chat_via_message_capable(&self) -> Option<&dyn chat_message::Backend> {
        Some(self)
    }
}

impl chat_message::Backend for Backend {
    fn instance(
        &self,
        reference: String,
        _config: ChatConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn chat_message::Instance>, BackendError>> + Send + '_>> {
        let handle = self.handle.clone();
        Box::pin(async move {
            let path = PathBuf::from(reference);
            let handle_for_load = handle.clone();
            tokio::task::spawn_blocking(move || handle_for_load.load(&path))
                .await
                .map_err(|error| -> BackendError {
                    Box::new(Error::LibraryLoad {
                        message: error.to_string(),
                    })
                })?
                .map_err(|error| -> BackendError { Box::new(error) })?;
            let size = handle.loaded_size();
            Ok(Box::new(Instance::new(handle, size)) as Box<dyn chat_message::Instance>)
        })
    }
}
