use std::{
    ffi::{CStr, CString, c_char},
    path::Path,
    ptr,
};

use libloading::Library;

use crate::error::Error;

pub const DEFAULT_BUFFER_SIZE: usize = 65536;
pub const RETRY_BUFFER_SIZE: usize = 256 * 1024;

#[derive(Clone, Copy)]
pub struct Ffi {
    pub load: unsafe extern "C" fn(*const u8, u64) -> i32,
    pub init: unsafe extern "C" fn(*const c_char, *const c_char, *const c_char) -> i32,
    pub complete: unsafe extern "C" fn(*const c_char, i32, *mut c_char, i32) -> i32,
    pub reset: unsafe extern "C" fn(),
}

pub struct Lib {
    _library: Option<Library>,
    pub ffi: Ffi,
}

impl Lib {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let library = unsafe { Library::new(path) }.map_err(|error| Error::LibraryLoad {
            message: format!("{}: {error}", path.display()),
        })?;
        let load = unsafe { symbol(&library, b"needle_load\0")? };
        let init = unsafe { symbol(&library, b"needle_init\0")? };
        let complete = unsafe { symbol(&library, b"needle_complete\0")? };
        let reset = unsafe { symbol(&library, b"needle_reset\0")? };
        Ok(Self {
            _library: Some(library),
            ffi: Ffi {
                load,
                init,
                complete,
                reset,
            },
        })
    }

    pub fn from_ffi(ffi: Ffi) -> Self {
        Self {
            _library: None,
            ffi,
        }
    }
}

unsafe fn symbol<T: Copy>(
    library: &Library,
    name: &[u8],
) -> Result<T, Error> {
    let symbol = unsafe { library.get::<T>(name) }.map_err(|_| Error::MissingSymbol {
        symbol: String::from_utf8_lossy(name).trim_end_matches('\0').to_string(),
    })?;
    Ok(*symbol)
}

pub fn cstring(text: &str) -> Result<CString, Error> {
    CString::new(text).map_err(|_| Error::CompleteFailed {
        code: -1,
        message: "needle input contains interior NUL".to_string(),
    })
}

pub fn complete_into(
    ffi: Ffi,
    input: &str,
    max_new_tokens: i32,
) -> Result<String, Error> {
    let c_input = cstring(input)?;
    let mut buf = vec![0u8; DEFAULT_BUFFER_SIZE];
    let rc =
        unsafe { (ffi.complete)(c_input.as_ptr(), max_new_tokens, buf.as_mut_ptr() as *mut c_char, buf.len() as i32) };
    if rc < 0 {
        buf = vec![0u8; RETRY_BUFFER_SIZE];
        let retry = unsafe {
            (ffi.complete)(c_input.as_ptr(), max_new_tokens, buf.as_mut_ptr() as *mut c_char, buf.len() as i32)
        };
        if retry < 0 {
            let detail = buffer_string(&buf);
            return Err(Error::CompleteFailed {
                code: retry,
                message: if detail.is_empty() {
                    "needle_complete failed".to_string()
                } else {
                    detail
                },
            });
        }
    }
    Ok(buffer_string(&buf))
}

pub fn init_with(
    ffi: Ffi,
    system: &str,
    tools_json: &str,
) -> Result<(), Error> {
    let system = cstring(system)?;
    let tools = cstring(tools_json)?;
    let rc = unsafe { (ffi.init)(system.as_ptr(), tools.as_ptr(), ptr::null()) };
    if rc < 0 {
        Err(Error::InitFailed)
    } else {
        Ok(())
    }
}

fn buffer_string(buf: &[u8]) -> String {
    CStr::from_bytes_until_nul(buf).ok().map(|value| value.to_string_lossy().into_owned()).unwrap_or_default()
}
