//! Regression tests for the JavaScript conversions that `bindings::export` generates for wasm.
//!
//! `wasm-bindgen-test-runner` runs them in Node:
//! `CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo test -p bindings-types
//! --features bindings-wasm --target wasm32-unknown-unknown --tests`.
#![cfg(target_arch = "wasm32")]

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use bindings_types::wasm::{FromJs, IntoJs, JsResult};
use js_sys::{Array, Function, JSON, Object, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::wasm_bindgen_test;

#[bindings::export(Structure(Class))]
#[derive(Clone)]
pub struct Record {
    pub name: String,
}

#[bindings::export(Structure(Class))]
#[derive(Clone)]
pub struct Other {
    pub name: String,
}

#[bindings::export(Enumeration)]
#[derive(Clone)]
pub enum Choice {
    Text {
        value: String,
    },
    Empty {},
}

const CLONE_RECORD: &str = "__uzuCloneRecord";

fn read_record(value: JsValue) -> JsResult<String> {
    Ok(Record::from_js(value)?.name)
}

fn read_choice(value: JsValue) -> JsResult<String> {
    Ok(match Choice::from_js(value)? {
        Choice::Text {
            value,
        } => value,
        Choice::Empty {} => String::new(),
    })
}

fn record(name: &str) -> JsValue {
    Record {
        name: name.to_string(),
    }
    .into_js()
}

/// Calls a JavaScript function with the comma-separated `params` bound to `args`.
fn js(
    params: &str,
    body: &str,
    args: &[&JsValue],
) -> JsValue {
    let args: Array = args.iter().copied().collect();
    Function::new_with_args(params, body).apply(&JsValue::UNDEFINED, &args).unwrap()
}

fn get(
    object: &JsValue,
    key: &str,
) -> JsValue {
    Reflect::get(object, &JsValue::from_str(key)).unwrap()
}

fn set(
    object: &JsValue,
    key: &str,
    value: &JsValue,
) -> JsResult<bool> {
    Reflect::set(object, &JsValue::from_str(key), value)
}

#[track_caller]
fn assert_type_error<T: std::fmt::Debug>(result: JsResult<T>) {
    let error = result.expect_err("expected a TypeError");
    assert!(error.is_instance_of::<js_sys::TypeError>(), "expected a TypeError, got {error:?}");
}

/// Checks `actual` is a plain object with the same own keys and values as `expected`.
#[track_caller]
fn assert_same_object(
    actual: &JsValue,
    expected: &JsValue,
) {
    assert_eq!(Object::get_prototype_of(actual), Object::get_prototype_of(&Object::new()));
    let keys = |object: &JsValue| {
        let mut keys: Vec<String> =
            Object::keys(object.unchecked_ref::<Object>()).iter().map(|key| key.as_string().unwrap()).collect();
        keys.sort();
        keys
    };
    assert_eq!(keys(actual), keys(expected));
    for key in keys(expected) {
        let json = |object: &JsValue| JSON::stringify(&get(object, &key)).unwrap();
        assert_eq!(json(actual), json(expected), "value of {key}");
    }
}

#[wasm_bindgen_test]
fn class_setter_rejects_wrong_type() {
    let record = record("original");
    assert_eq!(read_record(record.clone()).unwrap(), "original");
    assert_type_error(set(&record, "name", &JsValue::from(42)));
    assert_eq!(get(&record, "name"), "original");
    set(&record, "name", &JsValue::from("updated")).unwrap();
    assert_eq!(read_record(record).unwrap(), "updated");
}

#[wasm_bindgen_test]
fn variant_setter_rejects_wrong_type() {
    let choice = Choice::Text {
        value: "text".to_string(),
    }
    .into_js();
    assert_type_error(set(&choice, "value", &JsValue::from(42)));
    assert_eq!(get(&choice, "value"), "text");
    set(&choice, "value", &JsValue::from("changed")).unwrap();
    assert_eq!(read_choice(choice).unwrap(), "changed");
    assert_eq!(read_choice(Choice::Empty {}.into_js()).unwrap(), "");
}

#[wasm_bindgen_test]
fn forged_clone_tokens_are_rejected() {
    let invalid = [
        JsValue::from(0),
        JsValue::from(1),
        JsValue::from(-1),
        JsValue::from(f64::NAN),
        JsValue::from(f64::INFINITY),
        Object::new().into(),
        JsValue::NULL,
        JsValue::UNDEFINED,
    ];
    for token in &invalid {
        let forged = js("token", "return { __uzuCloneRecord: () => token }", &[token]);
        assert_type_error(read_record(forged));
    }
}

#[wasm_bindgen_test]
fn clone_of_another_class_is_rejected() {
    let other = Other {
        name: "wrong type".to_string(),
    }
    .into_js();
    let forged = js("other", "return { __uzuCloneRecord: () => other.__uzuCloneOther() }", &[&other]);
    assert_type_error(read_record(forged));
}

/// The runner loads a single module instance, so a value from a second instance can't be made here.
/// Its clone method would run outside any conversion of its own instance and return `undefined`,
/// which `forged_clone_tokens_are_rejected` covers.
#[wasm_bindgen_test]
fn clone_method_outside_conversion_returns_undefined() {
    let record = record("original");
    assert!(js("record", "return record.__uzuCloneRecord()", &[&record]).is_undefined());
}

#[wasm_bindgen_test]
fn clone_tokens_cannot_be_replayed() {
    let record = record("updated");
    let forged: Array = js(
        "record",
        "let stale;
        return [
            { __uzuCloneRecord() { stale = record.__uzuCloneRecord(); return stale; } },
            { __uzuCloneRecord: () => stale },
        ];",
        &[&record],
    )
    .unchecked_into();
    assert_eq!(read_record(forged.get(0)).unwrap(), "updated");
    assert_type_error(read_record(forged.get(1)));
}

#[wasm_bindgen_test]
fn throwing_clone_method_leaves_no_pending_clone() {
    let record = record("updated");
    let throwing = js(
        "record",
        "return { __uzuCloneRecord() { record.__uzuCloneRecord(); throw new Error('failed clone'); } }",
        &[&record],
    );
    let error: js_sys::Error = read_record(throwing).unwrap_err().unchecked_into();
    assert_eq!(String::from(error.message()), "failed clone");
    assert!(js("record", "return record.__uzuCloneRecord()", &[&record]).is_undefined());
    assert_eq!(read_record(record).unwrap(), "updated");
}

#[wasm_bindgen_test]
fn nested_conversions_use_separate_tokens() {
    let record = record("updated");
    let clone_record: Function = get(&record, CLONE_RECORD).unchecked_into();
    let nested = Rc::new(RefCell::new(None));
    let callback = {
        let (record, nested) = (record.clone(), nested.clone());
        Closure::<dyn FnMut() -> JsValue>::new(move || {
            let token = clone_record.call0(&record).unwrap();
            *nested.borrow_mut() = Some(read_record(record.clone()));
            token
        })
    };
    let reentrant = Object::new();
    set(&reentrant, CLONE_RECORD, callback.as_ref()).unwrap();
    assert_eq!(read_record(reentrant.into()).unwrap(), "updated");
    assert_eq!(nested.take().unwrap().unwrap(), "updated");
}

#[wasm_bindgen_test]
fn map_keys_named_like_prototype_members_stay_data() {
    let roundtrip = |value: &JsValue| HashMap::<String, JsValue>::from_js(value.clone()).unwrap().into_js();
    for value in [r#""text""#, r#"{ "nested": true }"#] {
        // JSON.parse defines "__proto__" as an own property instead of setting the prototype.
        let input = JSON::parse(&format!(r#"{{ "__proto__": {value}, "constructor": "ctor", "toString": "string" }}"#))
            .unwrap();
        let output = roundtrip(&input);
        assert_same_object(&output, &input);
        assert_same_object(&roundtrip(&output), &input);
    }
}
