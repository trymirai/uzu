use core::ptr::NonNull;

use objc2_core_foundation::{CFNumber, CFRetained, CFString, CFType, ConcreteType};

use super::{IOHIDEvent, IOHIDServiceClient, IOKit};

pub(super) struct ServiceClient {
    io_kit: &'static IOKit,
    inner: CFRetained<IOHIDServiceClient>,
}

impl From<(&'static IOKit, CFRetained<IOHIDServiceClient>)> for ServiceClient {
    fn from((io_kit, inner): (&'static IOKit, CFRetained<IOHIDServiceClient>)) -> Self {
        Self {
            io_kit,
            inner,
        }
    }
}

impl ServiceClient {
    fn property<T: ConcreteType>(
        &self,
        key: &str,
    ) -> Option<CFRetained<T>> {
        let key = CFString::from_str(key);
        let value = unsafe { (self.io_kit.copy_property)(&self.inner, &key) };
        let value: CFRetained<CFType> = unsafe { CFRetained::from_raw(NonNull::new(value)?) };
        value.downcast::<T>().ok()
    }

    pub(super) fn string(
        &self,
        key: &str,
    ) -> Option<String> {
        self.property::<CFString>(key).map(|string| string.to_string())
    }

    pub(super) fn i64_value(
        &self,
        key: &str,
    ) -> Option<i64> {
        self.property::<CFNumber>(key).and_then(|number| number.as_i64())
    }

    pub(super) fn f64_value(
        &self,
        event_type: i64,
        event_field: i32,
    ) -> Option<f64> {
        let event = unsafe { (self.io_kit.copy_event)(&self.inner, event_type, 0, 0) };
        let event: CFRetained<IOHIDEvent> = unsafe { CFRetained::from_raw(NonNull::new(event)?) };
        Some(unsafe { (self.io_kit.get_float_value)(&event, event_field) })
    }

    pub(super) fn registry_id(&self) -> u64 {
        unsafe { (self.io_kit.get_registry_id)(&self.inner) }
    }
}
