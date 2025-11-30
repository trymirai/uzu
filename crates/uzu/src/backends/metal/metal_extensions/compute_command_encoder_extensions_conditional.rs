#![allow(unexpected_cfgs)]

use metal::{
    BufferRef, ComputeCommandEncoder, ComputeCommandEncoderRef,
    MTLCompareFunction,
    foreign_types::{ForeignType, ForeignTypeRef},
    objc::{msg_send, runtime::Object, sel, sel_impl},
};

/// Invert a comparison to match Metal conditional encoding semantics.
fn invert_compare_fn(cmp: MTLCompareFunction) -> MTLCompareFunction {
    match cmp {
        MTLCompareFunction::Never => MTLCompareFunction::Always,
        MTLCompareFunction::Less => MTLCompareFunction::GreaterEqual,
        MTLCompareFunction::Equal => MTLCompareFunction::NotEqual,
        MTLCompareFunction::LessEqual => MTLCompareFunction::Greater,
        MTLCompareFunction::Greater => MTLCompareFunction::LessEqual,
        MTLCompareFunction::NotEqual => MTLCompareFunction::Equal,
        MTLCompareFunction::GreaterEqual => MTLCompareFunction::Less,
        MTLCompareFunction::Always => MTLCompareFunction::Never,
    }
}

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

    unsafe fn encode_start_while(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
    );

    unsafe fn encode_end_while(&self) -> bool;
}

/// Safe conditional wrapper that uses the raw unsafe methods internally.
pub trait ComputeEncoderConditional {
    fn condition<IfBlock, ElseBlock>(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
        if_block: IfBlock,
        else_block: Option<ElseBlock>,
    ) where
        IfBlock: FnOnce(),
        ElseBlock: FnOnce();

    fn while_loop<LoopBlock>(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
        loop_block: LoopBlock,
    ) where
        LoopBlock: FnMut();
}

impl ComputeEncoderRawConditional for ComputeCommandEncoder {
    unsafe fn encode_start_if(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
    ) {
        let obj = self.as_ptr() as *mut Object;
        let predicate_ptr = predicate.as_ptr() as *mut Object;
        let comparison = invert_compare_fn(comparison);
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

    unsafe fn encode_start_while(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
    ) {
        let obj = self.as_ptr() as *mut Object;
        let predicate_ptr = predicate.as_ptr() as *mut Object;
        let comparison = invert_compare_fn(comparison);
        let _: () = msg_send![
            obj,
            encodeStartWhile: predicate_ptr
            offset: offset
            comparison: comparison
            referenceValue: reference_value
        ];
    }

    unsafe fn encode_end_while(&self) -> bool {
        let obj = self.as_ptr() as *mut Object;
        let result: bool = msg_send![obj, encodeEndWhile];
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
        let obj = self as *const _ as *mut Object;
        let predicate_ptr = predicate.as_ptr() as *mut Object;
        let comparison = invert_compare_fn(comparison);
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

    unsafe fn encode_start_while(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
    ) {
        let obj = self as *const _ as *mut Object;
        let predicate_ptr = predicate.as_ptr() as *mut Object;
        let comparison = invert_compare_fn(comparison);
        let _: () = msg_send![
            obj,
            encodeStartWhile: predicate_ptr
            offset: offset
            comparison: comparison
            referenceValue: reference_value
        ];
    }

    unsafe fn encode_end_while(&self) -> bool {
        let obj = self as *const _ as *mut Object;
        let result: bool = msg_send![obj, encodeEndWhile];
        result
    }
}

impl<T> ComputeEncoderConditional for T
where
    T: ComputeEncoderRawConditional,
{
    fn condition<IfBlock, ElseBlock>(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
        if_block: IfBlock,
        else_block: Option<ElseBlock>,
    ) where
        IfBlock: FnOnce(),
        ElseBlock: FnOnce(),
    {
        match else_block {
            Some(else_fn) => unsafe {
                self.encode_start_if(
                    predicate,
                    offset,
                    comparison,
                    reference_value,
                );
                if_block();
                self.encode_start_else();
                else_fn();
                self.encode_end_if();
            },
            None => unsafe {
                self.encode_start_if(
                    predicate,
                    offset,
                    comparison,
                    reference_value,
                );
                if_block();
                self.encode_end_if();
            },
        }
    }

    fn while_loop<LoopBlock>(
        &self,
        predicate: &BufferRef,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
        mut loop_block: LoopBlock,
    ) where
        LoopBlock: FnMut(),
    {
        unsafe {
            self.encode_start_while(
                predicate,
                offset,
                comparison,
                reference_value,
            );
            loop_block();
            self.encode_end_while();
        }
    }
}
