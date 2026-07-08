use sysinfo::System;

pub(crate) fn build_system() -> System {
    let mut system = System::new_all();
    system.refresh_all();
    system
}
