use core::ptr::NonNull;

use obfstr::obfstr;
use objc2_core_foundation::{CFArray, CFDictionary, CFNumber, CFRetained, CFString};

use super::{IOHIDEventSystemClient, IOKit};

pub(super) struct EventSystemClient {
    io_kit: &'static IOKit,
    inner: CFRetained<IOHIDEventSystemClient>,
}

impl EventSystemClient {
    pub(super) fn new() -> Option<Self> {
        let io_kit = IOKit::get()?;
        let pointer = unsafe { (io_kit.create)(None) };
        let inner = unsafe { CFRetained::from_raw(NonNull::new(pointer)?) };
        Some(Self {
            io_kit,
            inner,
        })
    }

    pub(super) fn services_matching(
        &self,
        page: i32,
        usage: i32,
    ) -> Option<CFRetained<CFArray>> {
        let page_key = CFString::from_str(obfstr!("PrimaryUsagePage"));
        let usage_key = CFString::from_str(obfstr!("PrimaryUsage"));
        let page_value = CFNumber::new_i32(page);
        let usage_value = CFNumber::new_i32(usage);
        let matching = CFDictionary::from_slices(&[&*page_key, &*usage_key], &[&*page_value, &*usage_value]);

        let _ = unsafe { (self.io_kit.set_matching)(&self.inner, &matching) };

        let services = unsafe { (self.io_kit.copy_services)(&self.inner) };
        Some(unsafe { CFRetained::from_raw(NonNull::new(services)?) })
    }
}
