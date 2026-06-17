use core::{ffi::c_void, ptr::NonNull};

use objc2_core_foundation::{CFArray, CFDictionary, CFNumber, CFRetained, CFString, CFType};

use crate::{
    component::classify,
    sensor::{Sensor, SensorKind},
    sys::{self, Functions},
};

struct EventSystemClient {
    inner: CFRetained<CFType>,
}

impl EventSystemClient {
    fn new(functions: &Functions) -> Option<Self> {
        let pointer = unsafe { (functions.create)(core::ptr::null()) };
        let inner = unsafe { CFRetained::from_raw(NonNull::new(pointer.cast::<CFType>())?) };
        Some(Self {
            inner,
        })
    }

    fn as_raw_pointer(&self) -> *mut c_void {
        CFRetained::as_ptr(&self.inner).as_ptr().cast()
    }

    fn services_matching(
        &self,
        functions: &Functions,
        page: i32,
        usage: i32,
    ) -> Option<CFRetained<CFArray>> {
        let page_key = CFString::from_static_str("PrimaryUsagePage");
        let usage_key = CFString::from_static_str("PrimaryUsage");
        let page_value = CFNumber::new_i32(page);
        let usage_value = CFNumber::new_i32(usage);
        let matching = CFDictionary::from_slices(&[&*page_key, &*usage_key], &[&*page_value, &*usage_value]);

        let matching_pointer = CFRetained::as_ptr(&matching).as_ptr().cast::<c_void>().cast_const();
        let _ = unsafe { (functions.set_matching)(self.as_raw_pointer(), matching_pointer) };

        let services = unsafe { (functions.copy_services)(self.as_raw_pointer()) };
        let services = NonNull::new(services.cast_mut().cast::<CFArray>())?;
        Some(unsafe { CFRetained::from_raw(services) })
    }
}

fn service_string(
    functions: &Functions,
    service: *mut c_void,
    key: &str,
) -> Option<String> {
    let key = CFString::from_str(key);
    let key_pointer = CFRetained::as_ptr(&key).as_ptr().cast::<c_void>().cast_const();
    let value = unsafe { (functions.copy_property)(service, key_pointer) };
    let value = NonNull::new(value.cast_mut().cast::<CFType>())?;
    let value: CFRetained<CFType> = unsafe { CFRetained::from_raw(value) };
    value.downcast::<CFString>().ok().map(|string| string.to_string())
}

fn service_number(
    functions: &Functions,
    service: *mut c_void,
    key: &str,
) -> Option<i64> {
    let key = CFString::from_str(key);
    let key_pointer = CFRetained::as_ptr(&key).as_ptr().cast::<c_void>().cast_const();
    let value = unsafe { (functions.copy_property)(service, key_pointer) };
    let value = NonNull::new(value.cast_mut().cast::<CFType>())?;
    let value: CFRetained<CFType> = unsafe { CFRetained::from_raw(value) };
    value.downcast::<CFNumber>().ok().and_then(|number| number.as_i64())
}

fn service_float_value(
    functions: &Functions,
    service: *mut c_void,
    event_type: i64,
    event_field: i32,
) -> Option<f64> {
    let event = unsafe { (functions.copy_event)(service, event_type, 0, 0) };
    let event = NonNull::new(event.cast::<CFType>())?;
    let event: CFRetained<CFType> = unsafe { CFRetained::from_raw(event) };
    let value = unsafe { (functions.get_float_value)(CFRetained::as_ptr(&event).as_ptr().cast(), event_field) };
    Some(value)
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
        let service = unsafe { services.value_at_index(index) }.cast_mut();
        if service.is_null() {
            continue;
        }
        let Some(value) = service_float_value(functions, service, event_type, event_field) else {
            continue;
        };
        let name = service_string(functions, service, "Product").unwrap_or_default();
        let component = classify(&name);
        let registry_id = unsafe { (functions.get_registry_id)(service) };
        readings.push(Sensor {
            component,
            manufacturer: service_string(functions, service, "Manufacturer"),
            category: service_string(functions, service, "Category"),
            location_id: service_number(functions, service, "LocationID"),
            registry_id,
            name,
            value,
            kind,
        });
    }
    readings
}
