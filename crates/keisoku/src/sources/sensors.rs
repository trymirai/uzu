use crate::{
    sensor::{Sensor, SensorKind},
    sys::hid::{self, SensorReader},
};

pub(crate) fn collect(kind: SensorKind) -> Box<[Sensor]> {
    hid::collect(kind)
}

pub(crate) fn is_available() -> bool {
    hid::is_available()
}

pub(crate) fn new_reader(kind: SensorKind) -> Option<SensorReader> {
    SensorReader::new(kind)
}

pub(crate) fn read_reader(reader: &mut SensorReader) -> Box<[Sensor]> {
    reader.read()
}
