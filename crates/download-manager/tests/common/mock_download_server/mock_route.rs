use crate::common::mock_download_server::{FilePayload, RouteBehavior};

#[derive(Clone, Debug)]
pub(super) struct MockRoute {
    pub payload: FilePayload,
    pub behavior: RouteBehavior,
}
