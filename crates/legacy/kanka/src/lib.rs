//! `kanka` (甘果, "sweet fruit") — obfuscated `dlsym` bindings for private Apple frameworks.

#[cfg(target_vendor = "apple")]
pub use libc;
#[cfg(target_vendor = "apple")]
pub use obfstr;
#[cfg(target_vendor = "apple")]
pub use objc2_core_foundation;

/// Declares a struct of `dlsym`-resolved private C functions (names obfuscated),
/// with a `resolve()` and a `OnceLock`-cached `get()`.
#[macro_export]
macro_rules! ffi_table {
    (
        struct $table:ident from $framework:literal {
            $($field:ident = $symbol:literal: $signature:ty),* $(,)?
        }
    ) => {
        struct $table {
            $($field: $signature,)*
        }

        impl $table {
            fn resolve() -> ::core::option::Option<Self> {
                let path = ::std::ffi::CString::new($crate::obfstr::obfstr!($framework)).ok()?;
                let handle = unsafe { $crate::libc::dlopen(path.as_ptr(), $crate::libc::RTLD_LAZY) };
                if handle.is_null() {
                    return ::core::option::Option::None;
                }
                struct CloseOnDrop(*mut ::core::ffi::c_void);
                impl ::core::ops::Drop for CloseOnDrop {
                    fn drop(&mut self) {
                        unsafe { $crate::libc::dlclose(self.0); }
                    }
                }
                let guard = CloseOnDrop(handle);
                let table = Self {
                    $($field: {
                        let name = ::std::ffi::CString::new($crate::obfstr::obfstr!($symbol)).ok()?;
                        let symbol = unsafe { $crate::libc::dlsym(handle, name.as_ptr()) };
                        if symbol.is_null() {
                            return ::core::option::Option::None;
                        }
                        unsafe {
                            ::core::mem::transmute::<*mut ::core::ffi::c_void, $signature>(symbol)
                        }
                    },)*
                };
                ::core::mem::forget(guard);
                ::core::option::Option::Some(table)
            }

            fn get() -> ::core::option::Option<&'static Self> {
                static TABLE: ::std::sync::OnceLock<::core::option::Option<$table>> =
                    ::std::sync::OnceLock::new();
                TABLE.get_or_init(Self::resolve).as_ref()
            }
        }
    };
}

/// Declares an opaque CoreFoundation handle type usable with `CFRetained`.
#[macro_export]
macro_rules! opaque_cf_type {
    ($name:ident) => {
        #[repr(C)]
        #[allow(dead_code)]
        struct $name {
            inner: [u8; 0],
            _p: ::core::cell::UnsafeCell<
                ::core::marker::PhantomData<(
                    *const ::core::cell::UnsafeCell<()>,
                    ::core::marker::PhantomPinned,
                )>,
            >,
        }
        $crate::objc2_core_foundation::cf_type!(
            unsafe impl $name {}
        );
    };
}
