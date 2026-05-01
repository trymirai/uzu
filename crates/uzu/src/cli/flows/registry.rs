use std::{collections::HashMap, sync::Arc};

use super::Flow;

pub type FlowFactory = Arc<dyn Fn() -> Box<dyn Flow> + Send + Sync>;

#[derive(Clone, Default)]
pub struct FlowRegistry {
    factories: HashMap<String, FlowFactory>,
}

impl FlowRegistry {
    pub fn register<F>(
        mut self,
        name: &str,
        factory: F,
    ) -> Self
    where
        F: Fn() -> Box<dyn Flow> + Send + Sync + 'static,
    {
        self.factories.insert(name.to_string(), Arc::new(factory));
        self
    }

    pub fn create(
        &self,
        name: &str,
    ) -> Option<Box<dyn Flow>> {
        self.factories.get(name).map(|f| f())
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.factories.keys().map(|s| s.as_str())
    }
}
