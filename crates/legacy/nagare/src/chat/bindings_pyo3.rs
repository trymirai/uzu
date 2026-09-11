use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use pyo3::{Bound, Py, PyAny, PyErr, PyResult, Python, types::PyAnyMethods};
use shoji::types::basic::{ToolFunction, Value};

use super::ChatSession;
use crate::tool::func_def::{ErrorFuture, ToolDescriptor};

type PythonFuture = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>;

struct PythonInvocation {
    future: PythonFuture,
    cancellation: Option<Py<PyAny>>,
}

impl Future for PythonInvocation {
    type Output = PyResult<Py<PyAny>>;

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

impl Drop for PythonInvocation {
    fn drop(&mut self) {
        let Some(cancellation) = self.cancellation.take() else {
            return;
        };
        let _ = Python::try_attach(|py| {
            let _ = cancellation.bind(py).call_method0("cancel");
        });
    }
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pyo3::pymethods]
impl ChatSession {
    #[pyo3(name = "add_tool")]
    #[gen_stub(
        override_return_type(
            type_repr = "collections.abc.Awaitable[None]",
            imports = ("collections.abc")
        )
    )]
    fn add_tool_bindings_pyo3<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(
            override_type(
                type_repr = "UzuToolFunction[..., typing.Any]",
                imports = ("typing")
            )
        )]
        tool: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let descriptor = descriptor_from_python(py, tool)?;
        let mut session = self.clone();

        pyo3_async_runtimes::tokio::future_into_py(
            py,
            async move { session.add_tool(descriptor).await.map_err(Into::into) },
        )
    }

    #[pyo3(name = "add_tools")]
    #[gen_stub(
        override_return_type(
            type_repr = "collections.abc.Awaitable[None]",
            imports = ("collections.abc")
        )
    )]
    fn add_tools_bindings_pyo3<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(
            override_type(
                type_repr = "collections.abc.Sequence[UzuToolFunction[..., typing.Any]]",
                imports = ("collections.abc", "typing")
            )
        )]
        tools: Vec<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let descriptors =
            tools.into_iter().map(|tool| descriptor_from_python(py, tool)).collect::<PyResult<Vec<_>>>()?;
        let mut session = self.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            session.add_tools(descriptors).await.map_err(Into::into)
        })
    }
}

fn descriptor_from_python(
    py: Python<'_>,
    tool: Py<PyAny>,
) -> PyResult<ToolDescriptor> {
    let tool_function: ToolFunction = tool.bind(py).call_method0("_native_definition")?.extract()?;
    let tool = Arc::new(tool);
    Ok(ToolDescriptor::new(
        tool_function.name,
        tool_function.description,
        tool_function.parameters,
        tool_function.return_definition,
        Box::new(move |arguments| {
            let tool = Arc::clone(&tool);
            Box::new(call_python_tool(tool, arguments))
        }),
    ))
}

async fn call_python_tool(
    tool: Arc<Py<PyAny>>,
    arguments: Value,
) -> Result<Value, ErrorFuture> {
    let invocation = Python::attach(|py| -> PyResult<PythonInvocation> {
        let task_locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
        let cancellation = tool.bind(py).call_method1("_new_json_invocation", (arguments.json,))?;
        let awaitable = cancellation.call_method0("run")?;
        let future = Box::pin(pyo3_async_runtimes::into_future_with_locals(&task_locals, awaitable)?);
        Ok(PythonInvocation {
            future,
            cancellation: Some(cancellation.unbind()),
        })
    })
    .map_err(python_error)?;
    let result = invocation.await.map_err(python_error)?;

    let json = Python::attach(|py| result.extract::<String>(py)).map_err(python_error)?;
    let value = serde_json::from_str::<serde_json::Value>(&json)?;
    Ok(Value::from(value))
}

fn python_error(error: PyErr) -> ErrorFuture {
    error.to_string().into()
}

pub(super) fn spawn_with_current_task_locals<F>(future: F) -> tokio::task::JoinHandle<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    // `future_into_py` scopes the exported reply future with its Python loop and
    // context. Preserve that scope when the core session detaches the turn.
    let task_locals = Python::try_attach(|py| pyo3_async_runtimes::tokio::get_current_locals(py).ok()).flatten();
    match task_locals {
        Some(task_locals) => tokio::spawn(pyo3_async_runtimes::tokio::scope(task_locals, future)),
        None => tokio::spawn(future),
    }
}
