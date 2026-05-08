use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TerminalOutcome {
    Pending,
    Downloaded,
    Error(String),
    ActorStopped,
}
