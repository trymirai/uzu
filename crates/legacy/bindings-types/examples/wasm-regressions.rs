// Small runtime fixture exercising the generated bindings without engine dependencies.
use std::collections::HashMap;

use bindings_types::wasm::{FromJs, IntoJs, JsResult};
use wasm_bindgen::prelude::*;

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

#[wasm_bindgen]
pub fn read_record(value: JsValue) -> JsResult<String> {
    Ok(Record::from_js(value)?.name)
}

#[wasm_bindgen]
pub fn roundtrip_map(value: JsValue) -> JsResult<JsValue> {
    Ok(HashMap::<String, JsValue>::from_js(value)?.into_js())
}

#[bindings::export(Enumeration)]
#[derive(Clone)]
pub enum Choice {
    Text {
        value: String,
    },
    Empty {},
}

#[wasm_bindgen]
pub fn read_choice(value: JsValue) -> JsResult<String> {
    Ok(match Choice::from_js(value)? {
        Choice::Text {
            value,
        } => value,
        Choice::Empty {} => String::new(),
    })
}
