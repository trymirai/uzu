use std::{future::Future, pin::Pin};

use shoji::{traits::Registry, types::model::Model};

use crate::registry::RegistryError;

pub struct MergedRegistry {
    registries: Vec<Box<dyn Registry<Error = RegistryError>>>,
}

impl MergedRegistry {
    pub fn new(registries: Vec<Box<dyn Registry<Error = RegistryError>>>) -> Self {
        Self {
            registries,
        }
    }

    pub fn add(
        &mut self,
        registry: Box<dyn Registry<Error = RegistryError>>,
    ) -> Result<(), RegistryError> {
        if self.registries.iter().any(|current| current.indentifier() == registry.indentifier()) {
            return Err(RegistryError::UnableToAddRegistry {
                identifier: registry.indentifier(),
            });
        }
        self.registries.push(registry);
        Ok(())
    }

    pub fn remove(
        &mut self,
        identifier: &str,
    ) -> Option<(usize, Box<dyn Registry<Error = RegistryError>>)> {
        let index = self.registries.iter().position(|registry| registry.indentifier() == identifier)?;
        Some((index, self.registries.remove(index)))
    }

    pub fn restore(
        &mut self,
        index: usize,
        registry: Box<dyn Registry<Error = RegistryError>>,
    ) {
        self.registries.insert(index, registry);
    }
}

impl Registry for MergedRegistry {
    type Error = RegistryError;

    fn indentifier(&self) -> String {
        self.registries.iter().map(|registry| registry.indentifier()).collect::<Vec<String>>().join(":")
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
        Box::pin(async {
            let results = futures::future::join_all(self.registries.iter().map(|registry| registry.models())).await;

            let mut models = Vec::new();
            for result in results {
                models.extend(result?);
            }
            Ok(models)
        })
    }
}
