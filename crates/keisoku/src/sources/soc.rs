#[cfg(target_os = "macos")]
use crate::sys::soc::SocInfo;

#[cfg(target_os = "macos")]
pub(crate) fn new_soc() -> Option<SocInfo> {
    SocInfo::new()
}
