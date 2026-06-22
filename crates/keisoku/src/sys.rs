pub(crate) const HID_PAGE_APPLE_VENDOR: i32 = 0xff00;
pub(crate) const HID_PAGE_APPLE_VENDOR_POWER: i32 = 0xff08;
pub(crate) const HID_USAGE_TEMPERATURE_SENSOR: i32 = 0x0005;
pub(crate) const HID_USAGE_POWER_VOLTAGE: i32 = 0x0003;
pub(crate) const HID_USAGE_POWER_CURRENT: i32 = 0x0002;

pub(crate) const EVENT_TYPE_TEMPERATURE: i64 = 15;
pub(crate) const EVENT_TYPE_POWER: i64 = 25;

pub(crate) const fn event_field_base(event_type: i64) -> i32 {
    (event_type as i32) << 16
}
