use std::{ffi::CString, ptr::null_mut};

use anyhow::{Context, bail};
use shader_slang::{
    ComponentType, GenericArg, GenericArgType, Interface, Module, Session,
    reflection::{Decl, Generic, Shader, Type},
};
use shader_slang_sys::{IBlobVtable, ISlangBlob, ISlangUnknown__bindgen_vtable};

use super::Error;

unsafe fn blob_to_string(blob: *mut ISlangBlob) -> String {
    unsafe {
        let vtable = *(blob as *const *const IBlobVtable);
        let size = ((*vtable).getBufferSize)(blob as _);
        let buffer = ((*vtable).getBufferPointer)(blob as _) as *const u8;
        let message = if size == 0 {
            String::new()
        } else {
            String::from_utf8_lossy(std::slice::from_raw_parts(buffer, size)).into_owned()
        };
        let unknown = *(blob as *const *const ISlangUnknown__bindgen_vtable);
        ((*unknown).ISlangUnknown_release)(blob as _);
        message
    }
}
/// Loads a module, relaying its diagnostics as build warnings.
pub fn load_module(
    session: &Session,
    name: &str,
) -> Result<Module, Error> {
    let name_cstr = CString::new(name)?;
    let mut diagnostics: *mut ISlangBlob = null_mut();

    let module_ptr = unsafe { (session.vtable().loadModule)(session.as_raw(), name_cstr.as_ptr(), &mut diagnostics) };

    let diagnostics_str = if !diagnostics.is_null() {
        Some(unsafe { blob_to_string(diagnostics) })
    } else {
        None
    };

    if module_ptr.is_null() {
        if let Some(msg) = diagnostics_str {
            bail!("slang compilation failed: {}", msg);
        }
        bail!("slang compilation failed (no diagnostics)");
    }
    for line in diagnostics_str.iter().flat_map(|diagnostics| diagnostics.lines()) {
        println!("cargo::warning={line}");
    }

    unsafe {
        let vtable = *(module_ptr as *const *const ISlangUnknown__bindgen_vtable);
        ((*vtable).ISlangUnknown_addRef)(module_ptr as _);
    }

    let module: Module = unsafe { std::mem::transmute(std::ptr::NonNull::new(module_ptr).unwrap()) };
    Ok(module)
}
pub fn create_specialized_generic<'a>(
    component: &'a ComponentType,
    generic_decl: &Decl,
    concrete_types: &[&str],
) -> Result<&'a Generic, Error> {
    let layout: &Shader = component.layout(0)?;
    let generic: &Generic = generic_decl.as_generic().context("declaration has no generic reflection")?;

    let type_ptrs: Vec<&Type> = concrete_types
        .iter()
        .map(|name| layout.find_type_by_name(name).ok_or_else(|| anyhow::anyhow!("type '{}' not found", name)))
        .collect::<Result<_, Error>>()?;

    let arg_types: Vec<GenericArgType> =
        std::iter::repeat_n(GenericArgType::SlangGenericArgType, type_ptrs.len()).collect();
    let args: Vec<GenericArg> = type_ptrs
        .iter()
        .map(|t| GenericArg {
            typeVal: *t as *const _ as *mut _,
        })
        .collect();

    layout.specialize_generic(generic, &arg_types, &args).ok_or_else(|| anyhow::anyhow!("failed to specialize generic"))
}
