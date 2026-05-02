use std::{collections::HashMap, sync::Arc};

use crate::cli::flows::Flow;

pub type FlowFactory = Arc<dyn Fn() -> Box<dyn Flow> + Send + Sync>;

#[derive(Clone)]
pub struct Command {
    pub name: String,
    pub description: String,
    pub factory: FlowFactory,
}

#[derive(Clone, Default)]
pub struct FlowRegistry {
    commands: HashMap<String, Command>,
}

impl FlowRegistry {
    pub fn register<F>(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        factory: F,
    ) -> Self
    where
        F: Fn() -> Box<dyn Flow> + Send + Sync + 'static,
    {
        let name = name.into();
        let command = Command {
            name: name.clone(),
            description: description.into(),
            factory: Arc::new(factory),
        };
        self.commands.insert(name, command);
        self
    }

    pub fn create(
        &self,
        name: &str,
    ) -> Option<Box<dyn Flow>> {
        self.commands.get(name).map(|command| (command.factory)())
    }

    pub fn commands(&self) -> Vec<Command> {
        let mut commands: Vec<Command> = self.commands.values().cloned().collect();
        commands.sort_by(|first, second| first.name.cmp(&second.name));
        commands
    }
}
