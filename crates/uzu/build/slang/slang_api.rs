use std::{ffi::CString, ptr::null_mut};

use anyhow::bail;
use shader_slang::{
    ComponentType, Downcast, GenericArg, GenericArgType, Module, Session,
    reflection::{Decl, Generic, Shader, Type},
};
use shader_slang_sys::{
    IBlobVtable, ISessionVtable, ISlangBlob, ISlangUnknown__bindgen_vtable,
    spReflectionDecl_castToGeneric, spReflectionGeneric_GetTypeParameter,
    spReflectionGeneric_GetTypeParameterConstraintCount,
    spReflectionGeneric_GetTypeParameterConstraintType,
    spReflectionGeneric_GetTypeParameterCount, spReflectionType_GetName,
    spReflectionVariable_GetName,
};

unsafe fn blob_to_string(blob: *mut ISlangBlob) -> String {
    unsafe {
        let vtable = *(blob as *const *const IBlobVtable);
        let size = ((*vtable).getBufferSize)(blob as _);
        let buffer = ((*vtable).getBufferPointer)(blob as _) as *const u8;
        String::from_utf8_lossy(std::slice::from_raw_parts(buffer, size))
            .into_owned()
    }
}

pub struct ModuleWithDiagnostics {
    pub module: Module,
    pub diagnostics: Option<String>,
}

pub fn load_module(
    session: &Session,
    name: &str,
) -> anyhow::Result<ModuleWithDiagnostics> {
    let name_cstr = CString::new(name)?;
    let mut diagnostics: *mut ISlangBlob = null_mut();

    let module_ptr = unsafe {
        let session_ptr =
            std::mem::transmute_copy::<_, *mut std::ffi::c_void>(session);
        let vtable = *(session_ptr as *const *const ISessionVtable);
        ((*vtable).loadModule)(
            session_ptr,
            name_cstr.as_ptr(),
            &mut diagnostics,
        )
    };

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
        let vtable =
            *(module_ptr as *const *const ISlangUnknown__bindgen_vtable);
        ((*vtable).ISlangUnknown_addRef)(module_ptr as _);
    }

    let module = unsafe {
        std::mem::transmute(std::ptr::NonNull::new(module_ptr).unwrap())
    };

    Ok(ModuleWithDiagnostics {
        module,
        diagnostics: diagnostics_str,
    })
}

pub fn create_specialized_generic<'a>(
    module: &'a Module,
    generic_decl: &Decl,
    concrete_types: &[&str],
) -> anyhow::Result<&'a Generic> {
    let component: &ComponentType = module.downcast();
    let layout: &Shader = component.layout(0)?;
    let generic: &Generic = generic_decl.as_generic();

    let type_ptrs: Vec<&Type> = concrete_types
        .iter()
        .map(|name| {
            layout
                .find_type_by_name(name)
                .ok_or_else(|| anyhow::anyhow!("type '{}' not found", name))
        })
        .collect::<anyhow::Result<_>>()?;

    let arg_types: Vec<GenericArgType> =
        std::iter::repeat(GenericArgType::SlangGenericArgType)
            .take(type_ptrs.len())
            .collect();
    let args: Vec<GenericArg> = type_ptrs
        .iter()
        .map(|t| GenericArg {
            typeVal: *t as *const _ as *mut _,
        })
        .collect();

    layout
        .specialize_generic(generic, &arg_types, &args)
        .ok_or_else(|| anyhow::anyhow!("failed to specialize generic"))
}

pub struct TypeParameterInfo {
    pub name: String,
    pub constraints: Vec<String>,
}

pub fn get_generic_type_parameters(decl: &Decl) -> Vec<TypeParameterInfo> {
    unsafe {
        let generic_ptr =
            spReflectionDecl_castToGeneric(decl as *const _ as *mut _);
        if generic_ptr.is_null() {
            return Vec::new();
        }
        let count = spReflectionGeneric_GetTypeParameterCount(generic_ptr);
        (0..count)
            .filter_map(|i| {
                let type_param =
                    spReflectionGeneric_GetTypeParameter(generic_ptr, i);
                if type_param.is_null() {
                    return None;
                }
                let name_ptr = spReflectionVariable_GetName(type_param);
                if name_ptr.is_null() {
                    return None;
                }
                let name = std::ffi::CStr::from_ptr(name_ptr)
                    .to_string_lossy()
                    .into_owned();

                let constraint_count =
                    spReflectionGeneric_GetTypeParameterConstraintCount(
                        generic_ptr,
                        type_param,
                    );
                let constraints = (0..constraint_count)
                    .filter_map(|j| {
                        let constraint_type =
                            spReflectionGeneric_GetTypeParameterConstraintType(
                                generic_ptr,
                                type_param,
                                j,
                            );
                        if constraint_type.is_null() {
                            return None;
                        }
                        let constraint_name_ptr =
                            spReflectionType_GetName(constraint_type);
                        if constraint_name_ptr.is_null() {
                            return None;
                        }
                        Some(
                            std::ffi::CStr::from_ptr(constraint_name_ptr)
                                .to_string_lossy()
                                .into_owned(),
                        )
                    })
                    .collect();

                Some(TypeParameterInfo {
                    name,
                    constraints,
                })
            })
            .collect()
    }
}
