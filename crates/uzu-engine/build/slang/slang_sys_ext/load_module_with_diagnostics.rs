use std::{
    ffi::{CString, c_void},
    mem,
    path::Path,
    ptr::{NonNull, null_mut},
};

use anyhow::anyhow;
use shader_slang::{Blob, Interface, Module, Session};

pub trait SessionLoadModuleWithDiagnosticsExt {
    fn load_module_with_diagnostics(
        &self,
        path: &Path,
    ) -> anyhow::Result<(Module, Option<String>)>;
}

impl SessionLoadModuleWithDiagnosticsExt for Session {
    fn load_module_with_diagnostics(
        &self,
        path: &Path,
    ) -> anyhow::Result<(Module, Option<String>)> {
        let path_cstr = CString::new(path.to_str().unwrap()).unwrap();
        let mut diagnostics_ptr = null_mut();

        let module_ptr = vcall!(self, loadModule(path_cstr.as_ptr(), &mut diagnostics_ptr));

        let diagnostics = NonNull::new(diagnostics_ptr as *mut c_void)
            .map(|diagnostics_ptr_nn| -> Blob { unsafe { mem::transmute(diagnostics_ptr_nn) } })
            .map(|blob| blob.as_str().unwrap().to_owned());

        let Some(module_ptr_nn) = NonNull::new(module_ptr as *mut c_void) else {
            return Err(anyhow!("{}", diagnostics.as_deref().unwrap_or("<no diagnostics>")));
        };

        let module: Module = unsafe { mem::transmute(module_ptr_nn) };
        unsafe { (module.as_unknown().vtable().ISlangUnknown_addRef)(module.as_raw()) };

        Ok((module, diagnostics))
    }
}
