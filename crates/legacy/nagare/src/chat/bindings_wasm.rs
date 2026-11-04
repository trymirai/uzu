use bindings_types::{
    js_sys::{Error as JavaScriptError, Function, Promise},
    wasm::{FromJs, JsResult, clone_pointer, take_clone},
};
use kiban::rt::MakeSendFuture;
use shoji::types::basic::{ToolFunction, Value};
use uuid::Uuid;
use wasm_bindgen::{JsCast, prelude::*};
use wasm_bindgen_futures::JsFuture;

use super::ChatSession;
use crate::tool::func_def::{ErrorFuture, ToolDescriptor};

struct JavaScriptInvocation {
    cancel: Option<Function>,
    identifier: String,
}

impl Drop for JavaScriptInvocation {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.call1(&JsValue::UNDEFINED, &self.identifier.clone().into());
        }
    }
}

const NATIVE_TOOL_CLONE_METHOD: &str = "__uzuCloneNativeTool";

#[derive(Clone)]
#[wasm_bindgen]
pub struct NativeTool {
    descriptor: ToolDescriptor,
}

#[wasm_bindgen]
impl NativeTool {
    /// The callback receives the JSON arguments and an invocation ID, and returns JSON or a promise
    /// of JSON. `cancel` receives the invocation ID when the call is cancelled.
    #[wasm_bindgen(constructor)]
    pub fn new(
        #[wasm_bindgen(unchecked_param_type = "ToolFunction")] definition: JsValue,
        #[wasm_bindgen(unchecked_param_type = "(arg0: string, arg1: string) => Promise<string>")] invoke_json: Function,
        #[wasm_bindgen(unchecked_param_type = "(arg: string) => void")] cancel: Function,
    ) -> Result<Self, JsValue> {
        let definition = ToolFunction::from_js(definition)?;
        let descriptor = ToolDescriptor::new(
            definition.name,
            definition.description,
            definition.parameters,
            definition.return_definition,
            Box::new(move |arguments| {
                Box::new(MakeSendFuture::new(call_javascript_tool(invoke_json.clone(), cancel.clone(), arguments)))
            }),
        );
        Ok(Self {
            descriptor,
        })
    }

    #[wasm_bindgen(js_name = "__uzuCloneNativeTool", skip_typescript)]
    pub fn clone_pointer_bindings_wasm(&self) -> usize {
        clone_pointer(self)
    }
}

impl FromJs for NativeTool {
    fn from_js(value: JsValue) -> JsResult<Self> {
        take_clone(&value, NATIVE_TOOL_CLONE_METHOD, "NativeTool")
    }
}

#[wasm_bindgen]
impl ChatSession {
    #[wasm_bindgen(js_name = addTool)]
    pub async fn add_tool_bindings_wasm(
        &self,
        tool: &NativeTool,
    ) -> Result<(), JsValue> {
        let mut session = self.clone();
        session.add_tool(tool.descriptor.clone()).await.map_err(Into::into)
    }

    #[wasm_bindgen(js_name = addTools)]
    pub async fn add_tools_bindings_wasm(
        &self,
        #[wasm_bindgen(unchecked_param_type = "Array<NativeTool>")] tools: JsValue,
    ) -> Result<(), JsValue> {
        let tools = Vec::<NativeTool>::from_js(tools)?;
        let mut session = self.clone();
        session.add_tools(tools.into_iter().map(|tool| tool.descriptor).collect()).await.map_err(Into::into)
    }
}

async fn call_javascript_tool(
    invoke_json: Function,
    cancel: Function,
    arguments: Value,
) -> Result<Value, ErrorFuture> {
    let mut invocation = JavaScriptInvocation {
        cancel: Some(cancel),
        identifier: Uuid::new_v4().to_string(),
    };
    let result = async {
        let promise =
            invoke_json.call2(&JsValue::UNDEFINED, &arguments.json.into(), &invocation.identifier.clone().into())?;
        JsFuture::from(Promise::resolve(&promise)).await
    }
    .await;
    invocation.cancel = None;
    let json = result.map_err(javascript_error)?.as_string().ok_or("tool result must be a JSON string")?;
    Ok(serde_json::from_str::<serde_json::Value>(&json)?.into())
}

fn javascript_error(error: JsValue) -> ErrorFuture {
    error
        .dyn_ref::<JavaScriptError>()
        .map(|error| String::from(error.message()))
        .or_else(|| error.as_string())
        .unwrap_or_else(|| format!("{error:?}"))
        .into()
}
