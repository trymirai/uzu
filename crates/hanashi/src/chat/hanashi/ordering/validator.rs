use shoji::types::encoding::Role;

use crate::chat::hanashi::ordering::{Config, Error};

pub struct Validator {
    current: Option<Role>,
    config: Config,
}

impl Validator {
    pub fn new(config: Config) -> Self {
        Self {
            current: None,
            config,
        }
    }

    pub fn reset(&mut self) {
        self.current = None;
    }

    pub fn validate_next(
        &mut self,
        role: &Role,
    ) -> Result<(), Error> {
        if matches!(role, Role::Custom { .. }) {
            return Ok(());
        }

        match &self.current {
            None => {
                if !self.config.initial.contains(role) {
                    return Err(Error::InvalidInitial {
                        expected: format_roles(&self.config.initial),
                        got: role.clone(),
                    });
                }
            },
            Some(current) => {
                let allowed = self.config.transitions.get(current).ok_or_else(|| Error::NoTransitions {
                    role: current.clone(),
                })?;
                if !allowed.contains(role) {
                    return Err(Error::InvalidTransition {
                        after: current.to_string(),
                        expected: format_roles(allowed),
                        got: role.clone(),
                    });
                }
            },
        }
        self.current = Some(role.clone());
        Ok(())
    }
}

fn format_roles(roles: &[Role]) -> String {
    roles.iter().map(|role| role.to_string()).collect::<Vec<_>>().join(", ")
}
