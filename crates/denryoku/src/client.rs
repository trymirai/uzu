use core::ptr::NonNull;

use obfstr::obfstr;
use objc2_core_foundation::{CFArray, CFDictionary, CFNumber, CFRetained, CFString, CFType};

use crate::{
    component::classify,
    sensor::{Sensor, SensorKind},
    sys::{self, Functions, IOHIDEvent, IOHIDEventSystemClient, IOHIDServiceClient},
};

struct EventSystemClient {
    inner: CFRetained<IOHIDEventSystemClient>,
}

impl EventSystemClient {
    fn new(functions: &Functions) -> Option<Self> {
        let pointer = unsafe { (functions.create)(None) };
        let inner = unsafe { CFRetained::from_raw(NonNull::new(pointer)?) };
        Some(Self {
            inner,
        })
    }

    fn services_matching(
        &self,
        functions: &Functions,
        page: i32,
        usage: i32,
    ) -> Option<CFRetained<CFArray>> {
        let page_key = CFString::from_str(obfstr!("PrimaryUsagePage"));
        let usage_key = CFString::from_str(obfstr!("PrimaryUsage"));
        let page_value = CFNumber::new_i32(page);
        let usage_value = CFNumber::new_i32(usage);
        let matching = CFDictionary::from_slices(&[&*page_key, &*usage_key], &[&*page_value, &*usage_value]);

        let _ = unsafe { (functions.set_matching)(&self.inner, &matching) };

        let services = unsafe { (functions.copy_services)(&self.inner) };
        Some(unsafe { CFRetained::from_raw(NonNull::new(services)?) })
    }
}

fn service_string(
    functions: &Functions,
    service: &IOHIDServiceClient,
    key: &str,
) -> Option<String> {
    let key = CFString::from_str(key);
    let value = unsafe { (functions.copy_property)(service, &key) };
    let value: CFRetained<CFType> = unsafe { CFRetained::from_raw(NonNull::new(value)?) };
    value.downcast::<CFString>().ok().map(|string| string.to_string())
}

fn service_number(
    functions: &Functions,
    service: &IOHIDServiceClient,
    key: &str,
) -> Option<i64> {
    let key = CFString::from_str(key);
    let value = unsafe { (functions.copy_property)(service, &key) };
    let value: CFRetained<CFType> = unsafe { CFRetained::from_raw(NonNull::new(value)?) };
    value.downcast::<CFNumber>().ok().and_then(|number| number.as_i64())
}

fn service_float_value(
    functions: &Functions,
    service: &IOHIDServiceClient,
    event_type: i64,
    event_field: i32,
) -> Option<f64> {
    let event = unsafe { (functions.copy_event)(service, event_type, 0, 0) };
    let event: CFRetained<IOHIDEvent> = unsafe { CFRetained::from_raw(NonNull::new(event)?) };
    Some(unsafe { (functions.get_float_value)(&event, event_field) })
}

pub(crate) fn collect(kind: SensorKind) -> Vec<Sensor> {
    let Some(functions) = sys::functions() else {
        return Vec::new();
    };
    let Some(client) = EventSystemClient::new(functions) else {
        return Vec::new();
    };
    let (page, usage) = kind.matching();
    let Some(services) = client.services_matching(functions, page, usage) else {
        return Vec::new();
    };

    let event_type = kind.event_type();
    let event_field = sys::event_field_base(event_type);

    let count = services.count();
    let mut readings = Vec::with_capacity(count as usize);
    for index in 0..count {
        let pointer = unsafe { services.value_at_index(index) }.cast::<IOHIDServiceClient>();
        let Some(pointer) = NonNull::new(pointer.cast_mut()) else {
            continue;
        };
        // Borrowed from the array; valid for the duration of this iteration.
        let service = unsafe { pointer.as_ref() };
        let Some(value) = service_float_value(functions, service, event_type, event_field) else {
            continue;
        };
        let name = service_string(functions, service, obfstr!("Product")).unwrap_or_default();
        let component = classify(&name);
        let registry_id = unsafe { (functions.get_registry_id)(service) };
        readings.push(Sensor {
            component,
            manufacturer: service_string(functions, service, obfstr!("Manufacturer")),
            category: service_string(functions, service, obfstr!("Category")),
            location_id: service_number(functions, service, obfstr!("LocationID")),
            registry_id,
            name,
            value,
            kind,
        });
    }
    readings
}
