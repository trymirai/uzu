//! Newtype wrappers giving each reading a concrete unit, so a `Watts` can't be
//! mixed up with a `Percent` or a `Megahertz`. Each is `#[serde(transparent)]`,
//! so the JSON form is the bare number — the `Session` format is unchanged.

use serde::{Deserialize, Serialize};

macro_rules! unit {
    ($(#[$doc:meta])* $name:ident($inner:ty) = $suffix:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, Serialize, Deserialize)]
        #[serde(transparent)]
        #[repr(transparent)]
        pub struct $name(pub $inner);

        impl $name {
            /// The underlying scalar.
            pub const fn value(self) -> $inner {
                self.0
            }
        }

        impl core::fmt::Display for $name {
            fn fmt(
                &self,
                formatter: &mut core::fmt::Formatter<'_>,
            ) -> core::fmt::Result {
                write!(formatter, "{} {}", self.0, $suffix)
            }
        }

        impl From<$inner> for $name {
            fn from(value: $inner) -> Self {
                Self(value)
            }
        }
    };
}

unit!(/// Power, in watts.
    Watts(f32) = "W");
unit!(/// A utilization ratio, 0-100.
    Percent(f32) = "%");
unit!(/// Temperature, in degrees Celsius.
    Celsius(f32) = "°C");
unit!(/// Clock frequency, in megahertz.
    Megahertz(u32) = "MHz");
unit!(/// Memory bandwidth, in gigabytes per second.
    GigabytesPerSecond(f32) = "GB/s");
unit!(/// A byte count.
    Bytes(u64) = "B");
unit!(/// A duration, in milliseconds.
    Milliseconds(u64) = "ms");
unit!(/// Fan speed, in revolutions per minute.
    Rpm(f32) = "rpm");
