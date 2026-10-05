//! HIP runtime loaded at run time: the AMD driver ships it (`amdhip64_7.dll` on Windows,
//! `libamdhip64.so` with ROCm on Linux), so building uzu needs no HIP SDK.

use std::{
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    sync::OnceLock,
};

use libloading::Library;

use crate::backends::amdgpu::error::AmdgpuError;

pub type HipError = c_int;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipStream = *mut c_void;
pub type HipEvent = *mut c_void;

pub const HIP_SUCCESS: HipError = 0;

pub const HIP_LAUNCH_PARAM_BUFFER_POINTER: *mut c_void = 0x01 as *mut c_void;
pub const HIP_LAUNCH_PARAM_BUFFER_SIZE: *mut c_void = 0x02 as *mut c_void;
pub const HIP_LAUNCH_PARAM_END: *mut c_void = 0x03 as *mut c_void;

pub const HIP_MEMCPY_DEFAULT: c_int = 4;
pub const HIP_HOST_MALLOC_DEFAULT: c_uint = 0x0;
pub const HIP_HOST_MALLOC_NON_COHERENT: c_uint = 0x8000_0000;
pub const HIP_STREAM_NON_BLOCKING: c_uint = 0x1;
pub const HIP_EVENT_DEFAULT: c_uint = 0x0;
pub const HIP_EVENT_BLOCKING_SYNC: c_uint = 0x1;
pub const HIP_EVENT_DISABLE_TIMING: c_uint = 0x2;

macro_rules! hip_api {
    ($($name:ident: fn($($argument:ty),*) -> $result:ty;)*) => {
        #[allow(non_snake_case)]
        pub struct Hip {
            _library: Library,
            $(pub $name: unsafe extern "C" fn($($argument),*) -> $result,)*
        }

        impl Hip {
            #[allow(non_snake_case)]
            unsafe fn load(library: Library) -> Result<Self, libloading::Error> {
                unsafe {
                    $(let $name = *library.get::<unsafe extern "C" fn($($argument),*) -> $result>(
                        concat!(stringify!($name), "\0").as_bytes(),
                    )?;)*
                    Ok(Self { _library: library, $($name,)* })
                }
            }
        }
    };
}

hip_api! {
    hipInit: fn(c_uint) -> HipError;
    hipGetDeviceCount: fn(*mut c_int) -> HipError;
    hipSetDevice: fn(c_int) -> HipError;
    hipDeviceGetName: fn(*mut c_char, c_int, c_int) -> HipError;
    hipGetDevicePropertiesR0600: fn(*mut c_void, c_int) -> HipError;
    hipGetErrorString: fn(HipError) -> *const c_char;
    hipStreamCreateWithFlags: fn(*mut HipStream, c_uint) -> HipError;
    hipStreamDestroy: fn(HipStream) -> HipError;
    hipStreamSynchronize: fn(HipStream) -> HipError;
    hipDeviceSynchronize: fn() -> HipError;
    hipModuleLoadData: fn(*mut HipModule, *const c_void) -> HipError;
    hipModuleUnload: fn(HipModule) -> HipError;
    hipModuleGetFunction: fn(*mut HipFunction, HipModule, *const c_char) -> HipError;
    hipModuleLaunchKernel: fn(
        HipFunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, HipStream, *mut *mut c_void, *mut *mut c_void
    ) -> HipError;
    hipHostMalloc: fn(*mut *mut c_void, usize, c_uint) -> HipError;
    hipHostFree: fn(*mut c_void) -> HipError;
    hipMalloc: fn(*mut *mut c_void, usize) -> HipError;
    hipFree: fn(*mut c_void) -> HipError;
    hipMemcpyAsync: fn(*mut c_void, *const c_void, usize, c_int, HipStream) -> HipError;
    hipMemsetAsync: fn(*mut c_void, c_int, usize, HipStream) -> HipError;
    hipEventCreateWithFlags: fn(*mut HipEvent, c_uint) -> HipError;
    hipEventRecord: fn(HipEvent, HipStream) -> HipError;
    hipEventSynchronize: fn(HipEvent) -> HipError;
    hipEventQuery: fn(HipEvent) -> HipError;
    hipEventElapsedTime: fn(*mut f32, HipEvent, HipEvent) -> HipError;
    hipEventDestroy: fn(HipEvent) -> HipError;
}

const LIBRARY_NAMES: &[&str] = if cfg!(windows) {
    &["amdhip64_7.dll", "amdhip64_6.dll"]
} else {
    &["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"]
};

static HIP: OnceLock<Result<Hip, String>> = OnceLock::new();

/// The HIP runtime, loaded and initialized once per process. hipInit runs here, before any caller gets the
/// table: a first runtime call that initializes it implicitly (hipMalloc before any context) racing with hipInit
/// on other threads deadlocked HIP (the parallel test suite hung at start).
pub fn hip() -> Result<&'static Hip, AmdgpuError> {
    HIP.get_or_init(|| {
        let mut errors = Vec::new();
        for &name in LIBRARY_NAMES {
            match unsafe { Library::new(name) } {
                Ok(library) => match unsafe { Hip::load(library) } {
                    Ok(hip) => {
                        let status = unsafe { (hip.hipInit)(0) };
                        if status != HIP_SUCCESS {
                            return Err(format!("{name}: hipInit: {}", hip.error_string(status)));
                        }
                        return Ok(hip);
                    },
                    Err(error) => errors.push(format!("{name}: {error}")),
                },
                Err(error) => errors.push(format!("{name}: {error}")),
            }
        }
        Err(errors.join("; "))
    })
    .as_ref()
    .map_err(|message| AmdgpuError::RuntimeUnavailable(message.clone()))
}

impl Hip {
    pub fn error_string(
        &self,
        error: HipError,
    ) -> String {
        let message = unsafe { (self.hipGetErrorString)(error) };
        if message.is_null() {
            format!("HIP error {error}")
        } else {
            unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned()
        }
    }

    pub fn check(
        &self,
        call: &'static str,
        error: HipError,
    ) -> Result<(), AmdgpuError> {
        if error == HIP_SUCCESS {
            Ok(())
        } else {
            Err(AmdgpuError::Hip {
                call,
                code: error,
                message: self.error_string(error),
            })
        }
    }
}

/// `hip!(hip.hipFoo(args))` calls the function and turns a non-zero status into an `AmdgpuError`.
macro_rules! hip_call {
    ($hip:expr, $name:ident($($argument:expr),* $(,)?)) => {{
        let hip: &$crate::backends::amdgpu::hip::Hip = $hip;
        hip.check(stringify!($name), unsafe { (hip.$name)($($argument),*) })
    }};
}

pub(crate) use hip_call;
