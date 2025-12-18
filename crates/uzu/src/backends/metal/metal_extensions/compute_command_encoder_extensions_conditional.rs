use metal::{
    BufferRef, ComputeCommandEncoder, ComputeCommandEncoderRef,
    MTLCompareFunction,
    foreign_types::{ForeignType, ForeignTypeRef},
};
use objc2::{msg_send, runtime::AnyObject};

/// Low-level, unsafe conditional control of Metal encoders.
/// This is internal; users should prefer the safe `ComputeEncoderConditional::condition`.
trait ComputeEncoderRawConditional {
    unsafe fn encode_start_if(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
    );

    unsafe fn encode_start_else(&self);

    unsafe fn encode_end_if(&self) -> bool;
}

/// Safe conditional wrapper that uses the raw unsafe methods internally.
pub trait ComputeEncoderConditional {
    fn condition<IfBlock, ElseBlock>(
        &self,
        predicate: Option<&BufferRef>,
        if_block: IfBlock,
        else_block: Option<ElseBlock>,
    ) where
        IfBlock: FnOnce(),
        ElseBlock: FnOnce();
}

impl ComputeEncoderRawConditional for ComputeCommandEncoder {
    unsafe fn encode_start_if(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
    ) {
        let obj = self.as_ptr() as *mut AnyObject;
        let predicate_ptr = predicate.as_ptr() as *mut AnyObject;
        let _: () = msg_send![
            obj,
            encodeStartIf: predicate_ptr,
            offset: offset,
            comparison: comparison as u64,
            referenceValue: reference_value
        ];
    }

    unsafe fn encode_start_else(&self) {
        let obj = self.as_ptr() as *mut AnyObject;
        let _: () = msg_send![obj, encodeStartElse];
    }

    unsafe fn encode_end_if(&self) -> bool {
        let obj = self.as_ptr() as *mut AnyObject;
        let result: bool = msg_send![obj, encodeEndIf];
        result
    }
}

impl ComputeEncoderRawConditional for ComputeCommandEncoderRef {
    unsafe fn encode_start_if(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
    ) {
        let obj = self.as_ptr() as *mut AnyObject;
        let predicate_ptr = predicate.as_ptr() as *mut AnyObject;
        let _: () = msg_send![
            obj,
            encodeStartIf: predicate_ptr,
            offset: offset,
            comparison: comparison as u64,
            referenceValue: reference_value
        ];
    }

    unsafe fn encode_start_else(&self) {
        let obj = self.as_ptr() as *mut AnyObject;
        let _: () = msg_send![obj, encodeStartElse];
    }

    unsafe fn encode_end_if(&self) -> bool {
        let obj = self.as_ptr() as *mut AnyObject;
        let result: bool = msg_send![obj, encodeEndIf];
        result
    }
}

impl<T> ComputeEncoderConditional for T
where
    T: ComputeEncoderRawConditional,
{
    fn condition<IfBlock, ElseBlock>(
        &self,
        predicate: Option<&BufferRef>,
        if_block: IfBlock,
        else_block: Option<ElseBlock>,
    ) where
        IfBlock: FnOnce(),
        ElseBlock: FnOnce(),
    {
        match (predicate, else_block) {
            (Some(p), Some(else_fn)) => unsafe {
                self.encode_start_if(p, 0, MTLCompareFunction::Equal, 0);
                if_block();
                self.encode_start_else();
                else_fn();
                self.encode_end_if();
            },
            (Some(p), None) => unsafe {
                self.encode_start_if(p, 0, MTLCompareFunction::Equal, 0);
                if_block();
                self.encode_end_if();
            },
            (None, _) => {
                if_block();
            },
        }
    }
}
