use super::{role::Role, version::Version};

pub(super) struct ChatTurn {
    pub role: Role,
    pub versions: Vec<Version>,
    pub current: usize,
    pub reasoning_collapsed: bool,
}

impl ChatTurn {
    pub(super) fn user(text: String) -> Self {
        Self {
            role: Role::User,
            versions: vec![Version {
                text,
                ..Default::default()
            }],
            current: 0,
            reasoning_collapsed: false,
        }
    }

    pub(super) fn assistant(version: Version) -> Self {
        Self {
            role: Role::Assistant,
            versions: vec![version],
            current: 0,
            reasoning_collapsed: false,
        }
    }

    pub(super) fn cur(&self) -> &Version {
        &self.versions[self.current]
    }

    pub(super) fn cur_mut(&mut self) -> &mut Version {
        &mut self.versions[self.current]
    }
}
