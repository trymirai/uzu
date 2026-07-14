use crate::{
    sensor::{Sensor, SensorKind},
    sys::hid::SensorReader,
};

pub(crate) fn new_reader(kind: SensorKind) -> Option<SensorReader> {
    SensorReader::new(kind)
}

pub(crate) fn read_reader(reader: &mut SensorReader) -> Box<[Sensor]> {
    reader.read()
}
