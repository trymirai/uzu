use std::{future::Future, pin::Pin};

use crate::registry::{Error, Registry, types::Model};

pub struct MergedRegistry {
    registries: Vec<Box<dyn Registry>>,
}

impl MergedRegistry {
    pub fn new(registries: Vec<Box<dyn Registry>>) -> Self {
        Self {
            registries,
        }
    }

    pub fn add(
        &mut self,
        registry: Box<dyn Registry>,
    ) -> Result<(), Error> {
        if self.registries.iter().any(|current_registry| current_registry.indentifier() == registry.indentifier()) {
            return Err(Error::UnableToAddRegistry {
                identifier: registry.indentifier(),
            });
        }
        self.registries.push(registry);
        Ok(())
    }

    pub fn remove(
        &mut self,
        identifier: &str,
    ) -> Result<(), Error> {
        self.registries.retain(|registry| registry.indentifier() != identifier);
        Ok(())
    }
}

impl Registry for MergedRegistry {
    fn indentifier(&self) -> String {
        self.registries.iter().map(|registry| registry.indentifier()).collect::<Vec<String>>().join(":")
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, Error>> + Send + '_>> {
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
