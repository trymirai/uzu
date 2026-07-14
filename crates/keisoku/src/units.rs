use serde::{Deserialize, Serialize};

macro_rules! unit {
    (@shared $name:ident($inner:ty) = $suffix:literal) => {
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

        impl core::ops::Add for $name {
            type Output = Self;

            fn add(
                self,
                other: Self,
            ) -> Self {
                Self(self.0 + other.0)
            }
        }

        impl core::ops::Sub for $name {
            type Output = Self;

            fn sub(
                self,
                other: Self,
            ) -> Self {
                Self(self.0 - other.0)
            }
        }

        impl core::iter::Sum for $name {
            fn sum<I: Iterator<Item = Self>>(iterator: I) -> Self {
                Self(iterator.map(|unit| unit.0).sum())
            }
        }
    };

    (float $name:ident($inner:ty) = $suffix:literal) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, Serialize, Deserialize)]
        #[serde(transparent)]
        #[repr(transparent)]
        pub struct $name(pub $inner);

        unit!(@shared $name($inner) = $suffix);

        impl core::ops::Mul<$inner> for $name {
            type Output = Self;

            fn mul(
                self,
                factor: $inner,
            ) -> Self {
                Self(self.0 * factor)
            }
        }

        impl core::ops::Div<$inner> for $name {
            type Output = Self;

            fn div(
                self,
                divisor: $inner,
            ) -> Self {
                Self(self.0 / divisor)
            }
        }
    };

    (int $name:ident($inner:ty) = $suffix:literal) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        #[repr(transparent)]
        pub struct $name(pub $inner);

        unit!(@shared $name($inner) = $suffix);
    };
}

unit!(float Watts(f32) = "W");
unit!(float Joules(f32) = "J");
unit!(float Percent(f32) = "%");
unit!(float GigabytesPerSecond(f32) = "GB/s");
unit!(int Bytes(u64) = "B");
unit!(float Rpm(f32) = "rpm");
