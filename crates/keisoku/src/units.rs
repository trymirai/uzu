use serde::{Deserialize, Serialize};

macro_rules! unit {
    ($(#[$doc:meta])* $name:ident($inner:ty) = $suffix:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, Serialize, Deserialize)]
        #[serde(transparent)]
        #[repr(transparent)]
        pub struct $name(pub $inner);

        impl $name {

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

unit!(Watts(f32) = "W");
unit!(Joules(f32) = "J");
unit!(Percent(f32) = "%");
unit!(Celsius(f32) = "°C");
unit!(Megahertz(u32) = "MHz");
unit!(GigabytesPerSecond(f32) = "GB/s");
unit!(Bytes(u64) = "B");
unit!(Milliseconds(u64) = "ms");
unit!(Rpm(f32) = "rpm");
