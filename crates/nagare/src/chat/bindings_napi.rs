use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use napi::{
    Status,
    bindgen_prelude::{AsyncBlock, AsyncBlockBuilder, ClassInstance, Env, FnArgs, Function, Promise},
    threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode},
};
use shoji::types::basic::{ToolFunction, Value};
use uuid::Uuid;

use super::ChatSession;
use crate::tool::func_def::{ErrorFuture, ToolDescriptor};

type InvokeArguments = FnArgs<(String, String)>;
type InvokeFunction = ThreadsafeFunction<InvokeArguments, Promise<String>, InvokeArguments, Status, false, true>;
type CancelFunction = ThreadsafeFunction<String, (), String, Status, false, true>;
type JavaScriptFuture = Pin<Box<dyn Future<Output = napi::Result<String>> + Send>>;

struct JavaScriptInvocation {
    future: JavaScriptFuture,
    cancellation: Option<(Arc<CancelFunction>, String)>,
}

impl Future for JavaScriptInvocation {
    type Output = napi::Result<String>;

    fn poll(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Self::Output> {
        let result = self.future.as_mut().poll(context);
        if result.is_ready() {
            self.cancellation = None;
        }
        result
    }
}

impl Drop for JavaScriptInvocation {
    fn drop(&mut self) {
        let Some((cancel, invocation_id)) = self.cancellation.take() else {
            return;
        };
        let _ = cancel.call(invocation_id, ThreadsafeFunctionCallMode::NonBlocking);
    }
}

#[napi_derive::napi]
pub struct NativeTool {
    descriptor: ToolDescriptor,
}

#[napi_derive::napi]
impl NativeTool {
    #[napi(constructor)]
    pub fn new(
        definition: ToolFunction,
        invoke_json: Function<'_, FnArgs<(String, String)>, Promise<String>>,
        cancel: Function<'_, String, ()>,
    ) -> napi::Result<Self> {
        let invoke_json = Arc::new(invoke_json.build_threadsafe_function().weak::<true>().build()?);
        let cancel = Arc::new(cancel.build_threadsafe_function().weak::<true>().build()?);
        let descriptor = ToolDescriptor::new(
            definition.name,
            definition.description,
            definition.parameters,
            definition.return_definition,
            Box::new(move |arguments| {
                let invoke_json = invoke_json.clone();
                let cancel = cancel.clone();
                Box::new(call_javascript_tool(invoke_json, cancel, arguments))
            }),
        );
        Ok(Self {
            descriptor,
        })
    }
}

#[napi_derive::napi]
impl ChatSession {
    #[napi(js_name = "addTool")]
    pub fn add_tool_bindings_napi(
        &self,
        tool: ClassInstance<'_, NativeTool>,
        env: Env,
    ) -> napi::Result<AsyncBlock<()>> {
        let descriptor = tool.descriptor.clone();
        let mut session = self.clone();
        AsyncBlockBuilder::new(async move { session.add_tool(descriptor).await.map_err(Into::into) }).build(&env)
    }

    #[napi(js_name = "addTools")]
    pub fn add_tools_bindings_napi(
        &self,
        tools: Vec<ClassInstance<'_, NativeTool>>,
        env: Env,
    ) -> napi::Result<AsyncBlock<()>> {
        let descriptors = tools.iter().map(|tool| tool.descriptor.clone()).collect();
        let mut session = self.clone();
        AsyncBlockBuilder::new(async move { session.add_tools(descriptors).await.map_err(Into::into) }).build(&env)
    }
}

async fn call_javascript_tool(
    invoke_json: Arc<InvokeFunction>,
    cancel: Arc<CancelFunction>,
    arguments: Value,
) -> Result<Value, ErrorFuture> {
    let invocation_id = Uuid::new_v4().to_string();
    let invocation_id_for_call = invocation_id.clone();
    let future = Box::pin(async move {
        let arguments = InvokeArguments::from((arguments.json, invocation_id_for_call));
        let promise = invoke_json.call_async_catch(arguments).await?;
        promise.await
    });
    let invocation = JavaScriptInvocation {
        future,
        cancellation: Some((cancel, invocation_id)),
    };
    let json = invocation.await.map_err(javascript_error)?;
    let value = serde_json::from_str::<serde_json::Value>(&json)?;
    Ok(Value::from(value))
}

fn javascript_error(error: napi::Error) -> ErrorFuture {
    error.to_string().into()
}
