#[cfg(feature = "bindings-uniffi")]
uniffi::setup_scaffolding!();

#[cfg(feature = "bindings-pyo3")]
#[pyo3::pyfunction]
fn generate_annotations() -> pyo3::PyResult<()> {
    let annotations = pyo3_stub_gen::StubInfo::from_pyproject_toml("pyproject.toml")
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    annotations.generate().map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(())
}

#[cfg(feature = "bindings-pyo3")]
#[pyo3::pymodule]
fn uzu(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;

    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    pyo3_async_runtimes::tokio::init(builder);
    for entry in ::inventory::iter::<::bindings_types::PyClassRegistration>() {
        (entry.register)(m)?;
    }
    m.add_function(pyo3::wrap_pyfunction!(generate_annotations, m)?)?;
    Ok(())
}

#[cfg(feature = "bindings-pyo3")]
pyo3_stub_gen::define_stub_info_gatherer!(pyo3_bindings_annotations);

#[cfg(feature = "capability-cli")]
pub mod cli;
#[cfg(not(target_family = "wasm"))]
pub mod device;
#[cfg(not(target_family = "wasm"))]
pub mod engine;
#[cfg(not(target_family = "wasm"))]
pub mod helpers;
#[cfg(not(target_family = "wasm"))]
pub mod logs;
#[cfg(not(target_family = "wasm"))]
pub mod player;
#[cfg(not(target_family = "wasm"))]
pub mod registry;
#[cfg(not(target_family = "wasm"))]
pub mod settings;
#[cfg(not(target_family = "wasm"))]
pub mod storage;
#[cfg(not(target_family = "wasm"))]
pub use nagare as session;
#[cfg(not(target_family = "wasm"))]
pub use shoji::*;
