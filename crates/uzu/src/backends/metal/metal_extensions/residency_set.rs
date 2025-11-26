#![allow(unexpected_cfgs)]

use std::ffi::c_void;

use metal::{
    CommandQueueRef, DeviceRef, HeapRef,
    objc::{
        class, msg_send,
        runtime::Object,
        sel, sel_impl,
    },
};

pub struct ResidencySetDescriptor(*mut Object);

impl ResidencySetDescriptor {
    pub fn new() -> Self {
        unsafe {
            let cls = class!(MTLResidencySetDescriptor);
            let obj: *mut Object = msg_send![cls, alloc];
            let obj: *mut Object = msg_send![obj, init];
            Self(obj)
        }
    }

    pub fn set_label(&self, label: &str) {
        unsafe {
            let ns_string = create_nsstring(label);
            let _: () = msg_send![self.0, setLabel: ns_string];
        }
    }

    pub fn set_initial_capacity(&self, capacity: u64) {
        unsafe {
            let _: () = msg_send![self.0, setInitialCapacity: capacity];
        }
    }

    pub fn as_ptr(&self) -> *mut Object {
        self.0
    }
}

impl Drop for ResidencySetDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

pub struct ResidencySet(*mut Object);

impl ResidencySet {
    pub fn new(device: &DeviceRef, descriptor: &ResidencySetDescriptor) -> Option<Self> {
        unsafe {
            let mut error: *mut Object = std::ptr::null_mut();
            let obj: *mut Object = msg_send![
                device,
                newResidencySetWithDescriptor: descriptor.as_ptr()
                error: &mut error
            ];
            if obj.is_null() {
                if !error.is_null() {
                    let desc: *mut Object = msg_send![error, localizedDescription];
                    let cstr: *const i8 = msg_send![desc, UTF8String];
                    if !cstr.is_null() {
                        let err_str = std::ffi::CStr::from_ptr(cstr).to_string_lossy();
                        eprintln!("[ResidencySet] Creation failed: {}", err_str);
                    }
                }
                None
            } else {
                Some(Self(obj))
            }
        }
    }

    pub fn add_allocation(&self, heap: &HeapRef) {
        unsafe {
            let _: () = msg_send![self.0, addAllocation: heap];
        }
    }

    pub fn commit(&self) {
        unsafe {
            let _: () = msg_send![self.0, commit];
        }
    }

    pub fn request_residency(&self) {
        unsafe {
            let _: () = msg_send![self.0, requestResidency];
        }
    }

    pub fn end_residency(&self) {
        unsafe {
            let _: () = msg_send![self.0, endResidency];
        }
    }

    pub fn as_ptr(&self) -> *mut Object {
        self.0
    }
}

impl Drop for ResidencySet {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

pub trait CommandQueueResidencyExt {
    fn add_residency_set(&self, set: &ResidencySet);
    fn remove_residency_set(&self, set: &ResidencySet);
}

impl CommandQueueResidencyExt for CommandQueueRef {
    fn add_residency_set(&self, set: &ResidencySet) {
        unsafe {
            let _: () = msg_send![self, addResidencySet: set.as_ptr()];
        }
    }

    fn remove_residency_set(&self, set: &ResidencySet) {
        unsafe {
            let _: () = msg_send![self, removeResidencySet: set.as_ptr()];
        }
    }
}

unsafe fn create_nsstring(s: &str) -> *mut Object {
    let cls = class!(NSString);
    let bytes = s.as_ptr() as *const c_void;
    let len = s.len();
    let obj: *mut Object = msg_send![cls, alloc];
    msg_send![obj, initWithBytes:bytes length:len encoding:4u64] // NSUTF8StringEncoding = 4
}

