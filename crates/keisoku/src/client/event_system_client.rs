use core::ptr::NonNull;

use obfstr::obfstr;
use objc2_core_foundation::{CFArray, CFDictionary, CFNumber, CFRetained, CFString};

use super::{IOHIDEventSystemClient, IOHIDServiceClient, IOKit, service_client::ServiceClient};

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

    /// Match HID services for the given usage page/usage and return them as
    /// owning [`ServiceClient`]s. `CFArray::to_vec` retains each element, so the
    /// returned clients no longer depend on the matched array staying alive.
    pub(super) fn services(
        &self,
        page: i32,
        usage: i32,
    ) -> Box<[ServiceClient]> {
        let io_kit = self.io_kit;
        let Some(array) = self.services_matching(page, usage) else {
            return Box::default();
        };
        array.to_vec().into_iter().map(|inner| (io_kit, inner).into()).collect()
    }

    fn services_matching(
        &self,
        page: i32,
        usage: i32,
    ) -> Option<CFRetained<CFArray<IOHIDServiceClient>>> {
        let page_key = CFString::from_str(obfstr!("PrimaryUsagePage"));
        let usage_key = CFString::from_str(obfstr!("PrimaryUsage"));
        let page_value = CFNumber::new_i32(page);
        let usage_value = CFNumber::new_i32(usage);
        let matching = CFDictionary::from_slices(&[&*page_key, &*usage_key], &[&*page_value, &*usage_value]);

        let _ = unsafe { (self.io_kit.set_matching)(&self.inner, &matching) };

        // `CFArray`'s generic is `PhantomData`, so typing the matched array as
        // `CFArray<IOHIDServiceClient>` is a layout-sound pointer cast.
        let services = unsafe { (self.io_kit.copy_services)(&self.inner) };
        let services = NonNull::new(services.cast::<CFArray<IOHIDServiceClient>>())?;
        Some(unsafe { CFRetained::from_raw(services) })
    }
}
