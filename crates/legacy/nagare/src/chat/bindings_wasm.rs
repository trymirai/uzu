use bindings_types::js_sys::{Error as JavaScriptError, Function, Promise};
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

#[wasm_bindgen]
pub struct NativeTool {
    descriptor: ToolDescriptor,
}

#[wasm_bindgen]
impl NativeTool {
    /// The definition uses the serialized ToolFunction shape. The callback receives
    /// JSON arguments and an invocation ID, and returns JSON or a promise of JSON.
    #[wasm_bindgen(constructor)]
    pub fn new(
        definition_json: &str,
        invoke_json: Function,
        cancel: Function,
    ) -> Result<Self, JsValue> {
        let definition: ToolFunction =
            serde_json::from_str(definition_json).map_err(|error| JavaScriptError::new(&error.to_string()))?;
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

    /// Consumes the tool handles; the session retains their callbacks.
    #[wasm_bindgen(js_name = addTools)]
    pub async fn add_tools_bindings_wasm(
        &self,
        tools: Vec<NativeTool>,
    ) -> Result<(), JsValue> {
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
