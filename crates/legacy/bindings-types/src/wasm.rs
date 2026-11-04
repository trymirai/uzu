//! Conversions between Rust values and JavaScript values for the wasm bindings.
//!
//! `bindings::export` implements [`IntoJs`] and [`FromJs`] for every exported type, so that the
//! wasm package exposes the same JavaScript shapes as the napi package: camelCase fields, one class
//! per enum variant, string enums, `null` for a missing return value, and plain objects for maps.

use std::{collections::HashMap, hash::BuildHasher};

use js_sys::{Array, Function, Object, Reflect};
use wasm_bindgen::{JsCast, JsValue};

pub type JsResult<T> = Result<T, JsValue>;

pub trait IntoJs {
    fn into_js(self) -> JsValue;
}

pub trait FromJs: Sized {
    fn from_js(value: JsValue) -> JsResult<Self>;
}

/// The value a failed call rejects or throws with: an `Error` carrying the Rust error message.
pub fn error(message: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&message.to_string()).into()
}

pub fn type_error(message: impl std::fmt::Display) -> JsValue {
    js_sys::TypeError::new(&message.to_string()).into()
}

/// Moves a clone of an exported class value to the heap for [`take_clone`]. Each class exposes this
/// through a hidden method, so JavaScript objects passed to Rust are copied rather than consumed.
pub fn clone_pointer<T: Clone>(value: &T) -> usize {
    Box::into_raw(Box::new(value.clone())) as usize
}

/// Copies the Rust value out of a JavaScript object through its hidden clone method `method`.
///
/// `method` must be the name of a method generated for `T` alone that returns [`clone_pointer`].
pub fn take_clone<T>(
    value: &JsValue,
    method: &str,
    type_name: &str,
) -> JsResult<T> {
    let expected = || type_error(format!("expected {type_name}"));
    if !value.is_object() {
        return Err(expected());
    }
    let function: Function = Reflect::get(value, &JsValue::from_str(method))?.dyn_into().map_err(|_| expected())?;
    let pointer = function.call0(value)?.as_f64().ok_or_else(expected)? as usize;
    // SAFETY: `method` is unique to `T` and returns `Box::into_raw(Box<T>)`, see `clone_pointer`.
    Ok(unsafe { *Box::from_raw(pointer as *mut T) })
}

pub fn new_object() -> Object {
    Object::new()
}

pub fn set_property(
    object: &Object,
    key: &str,
    value: JsValue,
) {
    let _ = Reflect::set(object, &JsValue::from_str(key), &value);
}

pub fn expect_object(
    value: &JsValue,
    type_name: &str,
) -> JsResult<()> {
    if value.is_object() {
        Ok(())
    } else {
        Err(type_error(format!("expected {type_name}")))
    }
}

pub fn get_property(
    value: &JsValue,
    key: &str,
) -> JsResult<JsValue> {
    Reflect::get(value, &JsValue::from_str(key))
}

/// Reads the `type` tag of an error value, see `bindings::export(Error)`.
pub fn get_tag(
    value: &JsValue,
    type_name: &str,
) -> JsResult<String> {
    expect_object(value, type_name)?;
    get_property(value, "type")?.as_string().ok_or_else(|| type_error(format!("expected {type_name}")))
}

pub fn unknown_variant(
    type_name: &str,
    variant: &str,
) -> JsValue {
    type_error(format!("unknown {type_name} variant: {variant}"))
}

impl IntoJs for JsValue {
    fn into_js(self) -> JsValue {
        self
    }
}

impl FromJs for JsValue {
    fn from_js(value: JsValue) -> JsResult<Self> {
        Ok(value)
    }
}

impl IntoJs for () {
    fn into_js(self) -> JsValue {
        JsValue::UNDEFINED
    }
}

impl IntoJs for bool {
    fn into_js(self) -> JsValue {
        JsValue::from_bool(self)
    }
}

impl FromJs for bool {
    fn from_js(value: JsValue) -> JsResult<Self> {
        value.as_bool().ok_or_else(|| type_error("expected a boolean"))
    }
}

// napi passes every Rust integer as a JavaScript number, 64-bit ones included.
macro_rules! number {
    ($($number:ty),*) => {$(
        impl IntoJs for $number {
            fn into_js(self) -> JsValue {
                JsValue::from_f64(self as f64)
            }
        }

        impl FromJs for $number {
            fn from_js(value: JsValue) -> JsResult<Self> {
                value.as_f64().map(|number| number as $number).ok_or_else(|| type_error("expected a number"))
            }
        }
    )*};
}

number!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

impl IntoJs for String {
    fn into_js(self) -> JsValue {
        JsValue::from_str(&self)
    }
}

impl FromJs for String {
    fn from_js(value: JsValue) -> JsResult<Self> {
        value.as_string().ok_or_else(|| type_error("expected a string"))
    }
}

impl<T: IntoJs> IntoJs for Option<T> {
    fn into_js(self) -> JsValue {
        self.map_or(JsValue::NULL, IntoJs::into_js)
    }
}

impl<T: FromJs> FromJs for Option<T> {
    fn from_js(value: JsValue) -> JsResult<Self> {
        if value.is_null() || value.is_undefined() {
            Ok(None)
        } else {
            T::from_js(value).map(Some)
        }
    }
}

impl<T: IntoJs> IntoJs for Box<T> {
    fn into_js(self) -> JsValue {
        (*self).into_js()
    }
}

impl<T: FromJs> FromJs for Box<T> {
    fn from_js(value: JsValue) -> JsResult<Self> {
        T::from_js(value).map(Box::new)
    }
}

impl<T: IntoJs> IntoJs for Vec<T> {
    fn into_js(self) -> JsValue {
        self.into_iter().map(IntoJs::into_js).collect::<Array>().into()
    }
}

impl<T: FromJs> FromJs for Vec<T> {
    fn from_js(value: JsValue) -> JsResult<Self> {
        if !Array::is_array(&value) {
            return Err(type_error("expected an array"));
        }
        Array::from(&value).iter().map(T::from_js).collect()
    }
}

impl<V: IntoJs, S: BuildHasher> IntoJs for HashMap<String, V, S> {
    fn into_js(self) -> JsValue {
        let object = Object::new();
        for (key, value) in self {
            set_property(&object, &key, value.into_js());
        }
        object.into()
    }
}

impl<V: FromJs, S: BuildHasher + Default> FromJs for HashMap<String, V, S> {
    fn from_js(value: JsValue) -> JsResult<Self> {
        expect_object(&value, "an object")?;
        Object::entries(value.unchecked_ref())
            .iter()
            .map(|entry| {
                let entry = Array::from(&entry);
                Ok((String::from_js(entry.get(0))?, V::from_js(entry.get(1))?))
            })
            .collect()
    }
}
