#[cfg(feature = "bindings-pyo3")]
extern crate self as bindings_types;

#[cfg(feature = "bindings-pyo3")]
pub struct PyClassRegistration {
    pub register: fn(&::pyo3::Bound<'_, ::pyo3::types::PyModule>) -> ::pyo3::PyResult<()>,
}

#[cfg(feature = "bindings-pyo3")]
::inventory::collect!(PyClassRegistration);
