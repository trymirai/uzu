use std::{ffi::CString, ptr::null_mut};

use anyhow::{Context, bail};
use shader_slang::{
    ComponentType, GenericArg, GenericArgType, Interface, Module, Session,
    reflection::{Decl, Generic, Shader, Type},
};
use shader_slang_sys::{
    IBlobVtable, ISlangBlob, ISlangUnknown__bindgen_vtable, spReflectionDecl_castToGeneric,
    spReflectionGeneric_GetTypeParameter, spReflectionGeneric_GetTypeParameterConstraintCount,
    spReflectionGeneric_GetTypeParameterConstraintType, spReflectionGeneric_GetTypeParameterCount,
    spReflectionType_GetName, spReflectionVariable_GetName,
};

use super::{Error, ModuleWithDiagnostics, TypeParameterInfo};

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
pub fn load_module(
    session: &Session,
    name: &str,
) -> Result<ModuleWithDiagnostics, Error> {
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

    unsafe {
        let vtable = *(module_ptr as *const *const ISlangUnknown__bindgen_vtable);
        ((*vtable).ISlangUnknown_addRef)(module_ptr as _);
    }

    let module: Module = unsafe { std::mem::transmute(std::ptr::NonNull::new(module_ptr).unwrap()) };

    let component = module.clone().into();
    Ok(ModuleWithDiagnostics {
        module,
        component,
        diagnostics: diagnostics_str,
    })
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
pub fn get_generic_type_parameters(decl: &Decl) -> Result<Vec<TypeParameterInfo>, Error> {
    unsafe {
        let generic_ptr = spReflectionDecl_castToGeneric(decl as *const _ as *mut _);
        let generic_ptr =
            std::ptr::NonNull::new(generic_ptr).context("declaration has no generic reflection")?.as_ptr();
        let count = spReflectionGeneric_GetTypeParameterCount(generic_ptr);
        (0..count)
            .map(|i| {
                let type_param = spReflectionGeneric_GetTypeParameter(generic_ptr, i);
                let type_param = std::ptr::NonNull::new(type_param).context("generic has no type parameter")?.as_ptr();
                let name_ptr = spReflectionVariable_GetName(type_param);
                let name_ptr =
                    std::ptr::NonNull::new(name_ptr.cast_mut()).context("type parameter has no name")?.as_ptr();
                let name = std::ffi::CStr::from_ptr(name_ptr).to_str()?.to_string();

                let constraint_count = spReflectionGeneric_GetTypeParameterConstraintCount(generic_ptr, type_param);
                let constraints = (0..constraint_count)
                    .map(|j| {
                        let constraint_type =
                            spReflectionGeneric_GetTypeParameterConstraintType(generic_ptr, type_param, j);
                        let constraint_type = std::ptr::NonNull::new(constraint_type)
                            .context("type parameter has no constraint type")?
                            .as_ptr();
                        let constraint_name_ptr = spReflectionType_GetName(constraint_type);
                        let constraint_name_ptr = std::ptr::NonNull::new(constraint_name_ptr.cast_mut())
                            .context("constraint type has no name")?
                            .as_ptr();
                        Ok(std::ffi::CStr::from_ptr(constraint_name_ptr).to_str()?.to_string())
                    })
                    .collect::<Result<_, Error>>()?;

                Ok(TypeParameterInfo {
                    name,
                    constraints,
                })
            })
            .collect()
    }
}
