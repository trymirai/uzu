use std::{future::Future, pin::Pin, sync::Arc};

use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    types::{PyAnyMethods, PyString},
};
use shoji::types::basic::{ToolFunction, Value};

use super::ChatSession;
use crate::tool::func_def::{ErrorFuture, ToolDescriptor};

type PythonFuture = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>;

enum PythonCall {
    Ready(Py<PyAny>),
    Awaitable(PythonFuture),
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
        let task_locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
        let descriptor = descriptor_from_python(py, tool, task_locals)?;
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
        let task_locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
        let descriptors = tools
            .into_iter()
            .map(|tool| descriptor_from_python(py, tool, task_locals.clone()))
            .collect::<PyResult<Vec<_>>>()?;
        let mut session = self.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            session.add_tools(descriptors).await.map_err(Into::into)
        })
    }
}

fn descriptor_from_python(
    py: Python<'_>,
    tool: Py<PyAny>,
    task_locals: pyo3_async_runtimes::TaskLocals,
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
            let task_locals = task_locals.clone();
            Box::new(call_python_tool(tool, task_locals, arguments))
        }),
    ))
}

async fn call_python_tool(
    tool: Arc<Py<PyAny>>,
    task_locals: pyo3_async_runtimes::TaskLocals,
    arguments: Value,
) -> Result<Value, ErrorFuture> {
    let call = Python::attach(|py| -> PyResult<PythonCall> {
        let result = tool.bind(py).call_method1("_invoke_json", (arguments.json,))?;
        if result.is_instance_of::<PyString>() {
            return Ok(PythonCall::Ready(result.unbind()));
        }

        let inspect = pyo3::types::PyModule::import(py, "inspect")?;
        if inspect.call_method1("isawaitable", (&result,))?.is_truthy()? {
            let future = pyo3_async_runtimes::into_future_with_locals(&task_locals, result)?;
            Ok(PythonCall::Awaitable(Box::pin(future)))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "decorated tool invocation must return JSON text or an awaitable producing JSON text",
            ))
        }
    })
    .map_err(python_error)?;

    let result = match call {
        PythonCall::Ready(result) => result,
        PythonCall::Awaitable(future) => future.await.map_err(python_error)?,
    };

    let json = Python::attach(|py| result.extract::<String>(py)).map_err(python_error)?;
    let value = serde_json::from_str::<serde_json::Value>(&json)?;
    Ok(Value::from(value))
}

fn python_error(error: PyErr) -> ErrorFuture {
    error.to_string().into()
}
