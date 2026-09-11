use crate::backends::apple::AppleEventRegistry;

#[derive(Debug, Clone)]
pub struct AppleSessionDelegateIvars {
    pub event_registry: AppleEventRegistry,
}
