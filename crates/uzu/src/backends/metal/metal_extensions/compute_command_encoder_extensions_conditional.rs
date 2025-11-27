use metal::{
    BufferRef, ComputeCommandEncoder, ComputeCommandEncoderRef,
    MTLCompareFunction,
    foreign_types::{ForeignType, ForeignTypeRef},
    objc::{msg_send, runtime::Object, sel, sel_impl},
};

pub trait ComputeEncoderConditional {
    unsafe fn encode_start_if(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
    );

    unsafe fn encode_start_else(&self);

    unsafe fn encode_end_if(&self) -> bool;

    unsafe fn condition<F>(
        &self,
        predicate: Option<&BufferRef>,
        encoding_block: F,
    ) where
        F: FnOnce();
}

impl ComputeEncoderConditional for ComputeCommandEncoder {
    unsafe fn encode_start_if(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
    ) {
        let obj = self.as_ptr() as *mut Object;
        let predicate_ptr = predicate.as_ptr() as *mut Object;
        let _: () = msg_send![
            obj,
            encodeStartIf: predicate_ptr
            offset: offset
            comparison: comparison
            referenceValue: reference_value
        ];
    }

    unsafe fn encode_start_else(&self) {
        let obj = self.as_ptr() as *mut Object;
        let _: () = msg_send![obj, encodeStartElse];
    }

    unsafe fn encode_end_if(&self) -> bool {
        let obj = self.as_ptr() as *mut Object;
        let result: bool = msg_send![obj, encodeEndIf];
        result
    }

    unsafe fn condition<F>(
        &self,
        predicate: Option<&BufferRef>,
        encoding_block: F,
    ) where
        F: FnOnce(),
    {
        if let Some(p) = predicate {
            unsafe {
                self.encode_start_if(p, 0, MTLCompareFunction::NotEqual, 0);
            }
        }

        encoding_block();

        if let Some(_) = predicate {
            unsafe {
                self.encode_end_if();
            }
        }
    }
}

impl ComputeEncoderConditional for ComputeCommandEncoderRef {
    unsafe fn encode_start_if(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
    ) {
        let obj = self as *const _ as *mut Object;
        let predicate_ptr = predicate.as_ptr() as *mut Object;
        let _: () = msg_send![
            obj,
            encodeStartIf:predicate_ptr
            offset:offset
            comparison:comparison
            referenceValue:reference_value
        ];
    }

    unsafe fn encode_start_else(&self) {
        let obj = self as *const _ as *mut Object;
        let _: () = msg_send![obj, encodeStartElse];
    }

    unsafe fn encode_end_if(&self) -> bool {
        let obj = self as *const _ as *mut Object;
        let result: bool = msg_send![obj, encodeEndIf];
        result
    }

    unsafe fn condition<F>(
        &self,
        predicate: Option<&BufferRef>,
        encoding_block: F,
    ) where
        F: FnOnce(),
    {
        if let Some(p) = predicate {
            unsafe {
                self.encode_start_if(p, 0, MTLCompareFunction::NotEqual, 0);
            }
        }

        encoding_block();

        if let Some(_) = predicate {
            unsafe {
                self.encode_end_if();
            }
        }
    }
}
