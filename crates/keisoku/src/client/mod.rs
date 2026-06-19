use core::ptr::NonNull;

use obfstr::obfstr;
use objc2_core_foundation::{CFAllocator, CFArray, CFDictionary, CFNumber, CFRetained, CFString, CFType};

use self::{event_system_client::EventSystemClient, service_client::ServiceClient};
use crate::{
    component::{Component, classify},
    sensor::{Sensor, SensorKind},
    sys,
};

kanka::opaque_cf_type!(IOHIDEventSystemClient);
kanka::opaque_cf_type!(IOHIDServiceClient);
kanka::opaque_cf_type!(IOHIDEvent);

kanka::ffi_table! {
    struct IOKit from "/System/Library/Frameworks/IOKit.framework/IOKit" {
        create = "IOHIDEventSystemClientCreate":
            unsafe extern "C" fn(Option<&CFAllocator>) -> *mut IOHIDEventSystemClient,
        set_matching = "IOHIDEventSystemClientSetMatching":
            unsafe extern "C" fn(&IOHIDEventSystemClient, &CFDictionary<CFString, CFNumber>) -> i32,
        copy_services = "IOHIDEventSystemClientCopyServices":
            unsafe extern "C" fn(&IOHIDEventSystemClient) -> *mut CFArray,
        copy_property = "IOHIDServiceClientCopyProperty":
            unsafe extern "C" fn(&IOHIDServiceClient, &CFString) -> *mut CFType,
        copy_event = "IOHIDServiceClientCopyEvent":
            unsafe extern "C" fn(&IOHIDServiceClient, i64, i32, i64) -> *mut IOHIDEvent,
        get_float_value = "IOHIDEventGetFloatValue":
            unsafe extern "C" fn(&IOHIDEvent, i32) -> f64,
        get_registry_id = "IOHIDServiceClientGetRegistryID":
            unsafe extern "C" fn(&IOHIDServiceClient) -> u64,
    }
}

mod event_system_client;
mod service_client;

/// Whether the private IOKit HID API resolved on this system.
pub(crate) fn is_available() -> bool {
    IOKit::get().is_some()
}

/// Reads every sensor of `kind` in one full create→enumerate→read→release cycle.
///
/// The event-system client and its services live for the whole call, so reads
/// are always valid (matches macmon's `IOHIDSensors`); on return everything is
/// released together. No handles are retained across calls.
pub(crate) fn collect(kind: SensorKind) -> Vec<Sensor> {
    let Some(io_kit) = IOKit::get() else {
        return Vec::new();
    };
    let Some(client) = EventSystemClient::new() else {
        return Vec::new();
    };
    let (page, usage) = kind.matching();
    let Some(services) = client.services_matching(page, usage) else {
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
        let service = ServiceClient {
            io_kit,
            inner: unsafe { pointer.as_ref() },
        };
        let Some(value) = service.f64_value(event_type, event_field) else {
            continue;
        };
        let name = service.string(obfstr!("Product")).unwrap_or_default();
        readings.push(Sensor {
            component: classify(&name),
            manufacturer: service.string(obfstr!("Manufacturer")),
            category: service.string(obfstr!("Category")),
            location_id: service.i64_value(obfstr!("LocationID")),
            registry_id: service.registry_id(),
            name,
            value,
            kind,
        });
    }
    readings
}

/// The static (per-sensor, never-changing) metadata, read once at construction.
struct CachedSensor {
    service: NonNull<IOHIDServiceClient>,
    name: String,
    component: Component,
    manufacturer: Option<String>,
    category: Option<String>,
    location_id: Option<i64>,
    registry_id: u64,
}

/// A persistent sensor reader: enumerates the matching services and reads their
/// static metadata **once**, then every [`read`](Self::read) re-reads only the
/// live value per service. This avoids the per-sample client creation, service
/// enumeration, and four metadata IOKit round-trips per sensor that [`collect`]
/// repeats — the bulk of a sample's cost. The client and the services array are
/// both retained for the reader's lifetime, so the cached service pointers stay
/// valid (matching macmon's persistent `IOHIDSensors`); nothing is dropped early.
pub(crate) struct SensorReader {
    io_kit: &'static IOKit,
    kind: SensorKind,
    event_type: i64,
    event_field: i32,
    sensors: Vec<CachedSensor>,
    _services: CFRetained<CFArray>,
    _client: EventSystemClient,
}

impl SensorReader {
    pub(crate) fn new(kind: SensorKind) -> Option<Self> {
        let io_kit = IOKit::get()?;
        let client = EventSystemClient::new()?;
        let (page, usage) = kind.matching();
        let services = client.services_matching(page, usage)?;
        let event_type = kind.event_type();
        let event_field = sys::event_field_base(event_type);

        let count = services.count();
        let mut sensors = Vec::with_capacity(count as usize);
        for index in 0..count {
            let pointer = unsafe { services.value_at_index(index) }.cast::<IOHIDServiceClient>();
            let Some(pointer) = NonNull::new(pointer.cast_mut()) else {
                continue;
            };
            let service = ServiceClient {
                io_kit,
                inner: unsafe { pointer.as_ref() },
            };
            // Keep only services that actually report a value of this kind.
            if service.f64_value(event_type, event_field).is_none() {
                continue;
            }
            let name = service.string(obfstr!("Product")).unwrap_or_default();
            sensors.push(CachedSensor {
                component: classify(&name),
                manufacturer: service.string(obfstr!("Manufacturer")),
                category: service.string(obfstr!("Category")),
                location_id: service.i64_value(obfstr!("LocationID")),
                registry_id: service.registry_id(),
                service: pointer,
                name,
            });
        }

        Some(Self {
            io_kit,
            kind,
            event_type,
            event_field,
            sensors,
            _services: services,
            _client: client,
        })
    }

    /// Re-reads the live value of every cached sensor (one IOKit call each).
    pub(crate) fn read(&self) -> Vec<Sensor> {
        self.sensors
            .iter()
            .filter_map(|cached| {
                // Valid: the service pointer is owned by `_services`, kept alive
                // alongside `_client` for the whole lifetime of this reader.
                let service = ServiceClient {
                    io_kit: self.io_kit,
                    inner: unsafe { cached.service.as_ref() },
                };
                let value = service.f64_value(self.event_type, self.event_field)?;
                Some(Sensor {
                    component: cached.component,
                    manufacturer: cached.manufacturer.clone(),
                    category: cached.category.clone(),
                    location_id: cached.location_id,
                    registry_id: cached.registry_id,
                    name: cached.name.clone(),
                    value,
                    kind: self.kind,
                })
            })
            .collect()
    }
}
