use serde::{Deserialize, Serialize};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadManagerType {
    // NSURLSession. Downloads that carry a bearer token cannot resume here: pausing or quitting discards their
    // progress, see `AppleActiveTask::pause`.
    #[cfg(target_vendor = "apple")]
    #[default]
    Native,
    #[cfg_attr(not(target_vendor = "apple"), default)]
    Universal,
}
