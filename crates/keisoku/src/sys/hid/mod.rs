use std::time::{Duration, Instant};

use obfstr::obfstr;
use objc2_core_foundation::{CFAllocator, CFArray, CFDictionary, CFNumber, CFString, CFType};

use self::{event_system_client::EventSystemClient, service_client::ServiceClient};
use crate::{
    component::{Component, classify},
    sensor::{Sensor, SensorKind},
    sys,
};

const COLD_REFRESH: Duration = Duration::from_millis(3000);

fn is_hot(component: Component) -> bool {
    matches!(component, Component::Cpu | Component::Gpu | Component::Soc | Component::NeuralEngine)
}

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

pub(crate) fn is_available() -> bool {
    IOKit::get().is_some()
}

pub(crate) fn collect(kind: SensorKind) -> Box<[Sensor]> {
    let Some(client) = EventSystemClient::new() else {
        return Box::default();
    };
    let (page, usage) = kind.matching();
    let event_type = kind.event_type();
    let event_field = sys::event_field_base(event_type);

    let mut readings = Vec::new();
    for service in client.services(page, usage) {
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
    readings.into_boxed_slice()
}

struct CachedSensor {
    service: ServiceClient,
    name: String,
    component: Component,
    manufacturer: Option<String>,
    category: Option<String>,
    location_id: Option<i64>,
    registry_id: u64,
    hot: bool,
    value: f64,
}

pub(crate) struct SensorReader {
    kind: SensorKind,
    event_type: i64,
    event_field: i32,
    sensors: Vec<CachedSensor>,
    last_cold_read: Option<Instant>,
    _client: EventSystemClient,
}

impl SensorReader {
    pub(crate) fn new(kind: SensorKind) -> Option<Self> {
        let client = EventSystemClient::new()?;
        let (page, usage) = kind.matching();
        let event_type = kind.event_type();
        let event_field = sys::event_field_base(event_type);

        let mut sensors = Vec::new();
        for service in client.services(page, usage) {
            let Some(value) = service.f64_value(event_type, event_field) else {
                continue;
            };
            let name = service.string(obfstr!("Product")).unwrap_or_default();
            let component = classify(&name);
            sensors.push(CachedSensor {
                manufacturer: service.string(obfstr!("Manufacturer")),
                category: service.string(obfstr!("Category")),
                location_id: service.i64_value(obfstr!("LocationID")),
                registry_id: service.registry_id(),
                hot: is_hot(component) || matches!(kind, SensorKind::Voltage | SensorKind::Current),
                component,
                name,
                value,
                service,
            });
        }

        Some(Self {
            kind,
            event_type,
            event_field,
            sensors,
            last_cold_read: None,
            _client: client,
        })
    }

    pub(crate) fn read(&mut self) -> Box<[Sensor]> {
        let refresh_cold = self.last_cold_read.is_none_or(|at| at.elapsed() >= COLD_REFRESH);
        if refresh_cold {
            self.last_cold_read = Some(Instant::now());
        }
        let (kind, event_type, event_field) = (self.kind, self.event_type, self.event_field);
        self.sensors
            .iter_mut()
            .map(|cached| {
                let refreshed =
                    (cached.hot || refresh_cold).then(|| cached.service.f64_value(event_type, event_field)).flatten();
                if let Some(value) = refreshed {
                    cached.value = value;
                }
                Sensor {
                    component: cached.component,
                    manufacturer: cached.manufacturer.clone(),
                    category: cached.category.clone(),
                    location_id: cached.location_id,
                    registry_id: cached.registry_id,
                    name: cached.name.clone(),
                    value: cached.value,
                    kind,
                }
            })
            .collect()
    }
}
