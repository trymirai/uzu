use monostate::{ConstStr, MustBeBool, MustBeStr};
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum Unsupported {}

pub trait DeserializeStrict<'de>: Deserialize<'de> {}
pub trait DeserializeStrictOwned: for<'de> DeserializeStrict<'de> {}
impl<T: for<'de> DeserializeStrict<'de>> DeserializeStrictOwned for T {}

macro_rules! impl_strict {
  ($($ty:ty),* $(,)?) => {
      $(impl<'de> DeserializeStrict<'de> for $ty {})*
  };
}

impl_strict!(Unsupported, String, f32, u32, u64, usize, bool);

impl<'de, T: DeserializeStrict<'de>> DeserializeStrict<'de> for Box<T> {}
impl<'de, T: DeserializeStrict<'de>> DeserializeStrict<'de> for Box<[T]> {}
impl<'de, T: DeserializeStrict<'de>> DeserializeStrict<'de> for Option<T> {}
impl<'de, A: DeserializeStrict<'de>, B: DeserializeStrict<'de>> DeserializeStrict<'de> for (A, B) {}
impl<'de, const V: bool> DeserializeStrict<'de> for MustBeBool<V> {}
impl<'de, V: ConstStr> DeserializeStrict<'de> for MustBeStr<V> {}

pub fn required<'de, D: Deserializer<'de>, T: DeserializeStrict<'de>>(deserializer: D) -> Result<T, D::Error> {
    T::deserialize(deserializer)
}
