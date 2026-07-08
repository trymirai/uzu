use sysinfo::System;

pub(crate) fn build_system() -> System {
    let mut system = System::new_all();
    system.refresh_all();
    system
}

pub(crate) fn os_version(_system: &System) -> String {
    System::long_os_version().unwrap_or_default()
}
