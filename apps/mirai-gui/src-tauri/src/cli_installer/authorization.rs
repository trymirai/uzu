use std::{
    ffi::CString,
    os::raw::{c_char, c_int, c_void},
};

type AuthorizationRef = *mut c_void;
type CFTypeRef = *const c_void;
type CFStringRef = *const c_void;

#[repr(C)]
struct AuthorizationItem {
    name: *const c_char,
    value_length: usize,
    value: *mut c_void,
    flags: u32,
}

#[repr(C)]
struct AuthorizationItemSet {
    count: u32,
    items: *mut AuthorizationItem,
}

const ERR_SUCCESS: i32 = 0;
const ERR_CANCELED: i32 = -60006;
const ERR_DENIED: i32 = -60005;

const FLAG_DEFAULTS: u32 = 0;
const FLAG_INTERACTION_ALLOWED: u32 = 1 << 0;
const FLAG_EXTEND_RIGHTS: u32 = 1 << 1;
const FLAG_DESTROY_RIGHTS: u32 = 1 << 3;

const K_CFSTRING_ENCODING_UTF8: u32 = 0x0800_0100;

#[link(name = "Security", kind = "framework")]
unsafe extern "C" {
    fn AuthorizationCreate(
        rights: *const AuthorizationItemSet,
        environment: *const AuthorizationItemSet,
        flags: u32,
        authorization: *mut AuthorizationRef,
    ) -> i32;
    fn AuthorizationRightGet(
        right_name: *const c_char,
        right_definition: *mut CFTypeRef,
    ) -> i32;
    fn AuthorizationRightSet(
        auth_ref: AuthorizationRef,
        right_name: *const c_char,
        right_definition: CFTypeRef,
        description_key: CFStringRef,
        bundle: *const c_void,
        locale_table_name: CFStringRef,
    ) -> i32;
    fn AuthorizationCopyRights(
        authorization: AuthorizationRef,
        rights: *const AuthorizationItemSet,
        environment: *const AuthorizationItemSet,
        flags: u32,
        authorized_rights: *mut *mut AuthorizationItemSet,
    ) -> i32;
    fn AuthorizationExecuteWithPrivileges(
        authorization: AuthorizationRef,
        path_to_tool: *const c_char,
        options: u32,
        arguments: *const *const c_char,
        communications_pipe: *mut *mut c_void,
    ) -> i32;
    fn AuthorizationFree(
        authorization: AuthorizationRef,
        flags: u32,
    ) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(
        alloc: *const c_void,
        c_str: *const c_char,
        encoding: u32,
    ) -> CFStringRef;
    fn CFRelease(cf: CFTypeRef);
}

unsafe extern "C" {
    fn fread(
        ptr: *mut c_void,
        size: usize,
        nitems: usize,
        stream: *mut c_void,
    ) -> usize;
    fn fclose(stream: *mut c_void) -> c_int;
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    // cli_install turns this into the Cancelled result instead of an error.
    #[error("cancelled")]
    Cancelled,
    #[error("{0}")]
    Other(String),
}

pub fn run_privileged(
    bundle_id: &str,
    right_suffix: &str,
    prompt: &str,
    sh_command: &str,
) -> Result<(), AuthError> {
    let right_name =
        CString::new(format!("{bundle_id}.{right_suffix}")).map_err(|e| AuthError::Other(e.to_string()))?;
    let rule = CString::new("authenticate-admin").unwrap();
    let prompt_c = CString::new(prompt).map_err(|e| AuthError::Other(e.to_string()))?;
    let shell = CString::new("/bin/sh").unwrap();
    let flag_c = CString::new("-c").unwrap();
    let cmd_c = CString::new(sh_command).map_err(|e| AuthError::Other(e.to_string()))?;

    unsafe {
        let mut auth: AuthorizationRef = std::ptr::null_mut();
        if AuthorizationCreate(std::ptr::null(), std::ptr::null(), FLAG_DEFAULTS, &mut auth) != ERR_SUCCESS {
            return Err(AuthError::Other("AuthorizationCreate failed".into()));
        }
        let free = |auth: AuthorizationRef| {
            AuthorizationFree(auth, FLAG_DESTROY_RIGHTS);
        };

        // Register the right only when absent to avoid churning the policy database.
        if AuthorizationRightGet(right_name.as_ptr(), std::ptr::null_mut()) == ERR_DENIED {
            let rule_cf = CFStringCreateWithCString(std::ptr::null(), rule.as_ptr(), K_CFSTRING_ENCODING_UTF8);
            let prompt_cf = CFStringCreateWithCString(std::ptr::null(), prompt_c.as_ptr(), K_CFSTRING_ENCODING_UTF8);
            AuthorizationRightSet(auth, right_name.as_ptr(), rule_cf, prompt_cf, std::ptr::null(), std::ptr::null());
            CFRelease(rule_cf);
            CFRelease(prompt_cf);
        }

        let mut right_item = AuthorizationItem {
            name: right_name.as_ptr(),
            value_length: 0,
            value: std::ptr::null_mut(),
            flags: 0,
        };
        let rights = AuthorizationItemSet {
            count: 1,
            items: &mut right_item,
        };
        let status = AuthorizationCopyRights(
            auth,
            &rights,
            std::ptr::null(),
            FLAG_EXTEND_RIGHTS | FLAG_INTERACTION_ALLOWED,
            std::ptr::null_mut(),
        );
        if status != ERR_SUCCESS {
            free(auth);
            return Err(if status == ERR_CANCELED {
                AuthError::Cancelled
            } else {
                AuthError::Other(format!("AuthorizationCopyRights failed: {status}"))
            });
        }

        let argv: [*const c_char; 3] = [flag_c.as_ptr(), cmd_c.as_ptr(), std::ptr::null()];
        let mut pipe: *mut c_void = std::ptr::null_mut();
        let status = AuthorizationExecuteWithPrivileges(auth, shell.as_ptr(), FLAG_DEFAULTS, argv.as_ptr(), &mut pipe);
        if status != ERR_SUCCESS {
            free(auth);
            return Err(if status == ERR_CANCELED {
                AuthError::Cancelled
            } else {
                AuthError::Other(format!("AuthorizationExecuteWithPrivileges failed: {status}"))
            });
        }

        // Drain to EOF: blocks until the privileged /bin/sh finishes
        // without reaping unrelated children.
        if !pipe.is_null() {
            let mut buf = [0u8; 256];
            while fread(buf.as_mut_ptr().cast(), 1, buf.len(), pipe) > 0 {}
            fclose(pipe);
        }
        free(auth);
    }
    Ok(())
}
