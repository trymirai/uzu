use core::ptr::NonNull;

use obfstr::obfstr;
use objc2_core_foundation::CFRetained;

use super::{IOHIDServiceClient, IOKit, event_system_client::EventSystemClient, service_client::ServiceClient};
use crate::{
    component::classify,
    sensor::{Sensor, SensorKind},
    sys,
};

/// Enumerates the sensors of a kind once, retaining their service handles and
/// caching their static descriptors, then re-reads only the live value on each
/// [`sample`](Sampler::sample) — avoiding re-enumeration and metadata reads.
pub struct Sampler {
    io_kit: &'static IOKit,
    event_type: i64,
    event_field: i32,
    services: Vec<CFRetained<IOHIDServiceClient>>,
    readings: Vec<Sensor>,
}

impl Sampler {
    pub fn new(kind: SensorKind) -> Option<Self> {
        let io_kit = IOKit::get()?;
        let client = EventSystemClient::new()?;
        let (page, usage) = kind.matching();
        let services = client.services_matching(page, usage)?;

        let event_type = kind.event_type();
        let event_field = sys::event_field_base(event_type);

        let count = services.count();
        let mut handles = Vec::with_capacity(count as usize);
        let mut readings = Vec::with_capacity(count as usize);
        for index in 0..count {
            let pointer = unsafe { services.value_at_index(index) }.cast::<IOHIDServiceClient>();
            let Some(pointer) = NonNull::new(pointer.cast_mut()) else {
                continue;
            };
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
            // Retain the service so it outlives `client`/`services`, which drop
            // at the end of this function; the live value is re-read from it on
            // each `sample`.
            handles.push(unsafe { CFRetained::retain(pointer) });
        }

        Some(Self {
            io_kit,
            event_type,
            event_field,
            services: handles,
            readings,
        })
    }

    pub fn sample(&mut self) -> &[Sensor] {
        let io_kit = self.io_kit;
        let (event_type, event_field) = (self.event_type, self.event_field);
        for (handle, reading) in self.services.iter().zip(self.readings.iter_mut()) {
            let service = ServiceClient {
                io_kit,
                inner: handle,
            };
            if let Some(value) = service.f64_value(event_type, event_field) {
                reading.value = value;
            }
        }
        &self.readings
    }
}

pub(crate) fn collect(kind: SensorKind) -> Vec<Sensor> {
    Sampler::new(kind).map(|sampler| sampler.readings).unwrap_or_default()
}
