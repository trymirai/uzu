use objc2::msg_send;

use crate::backends::metal::{
    MTLBuffer, MTLCompareFunction, MTLComputeCommandEncoder, ProtocolObject,
};

/// Low-level, unsafe conditional control of Metal encoders.
/// This is internal; users should prefer the safe `ComputeEncoderConditional::condition`.
trait ComputeEncoderRawConditional {
    unsafe fn encode_start_if(
        &self,
        predicate: &ProtocolObject<dyn MTLBuffer>,
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
        predicate: Option<&ProtocolObject<dyn MTLBuffer>>,
        if_block: IfBlock,
        else_block: Option<ElseBlock>,
    ) where
        IfBlock: FnOnce(),
        ElseBlock: FnOnce();
}

impl ComputeEncoderRawConditional
    for ProtocolObject<dyn MTLComputeCommandEncoder>
{
    unsafe fn encode_start_if(
        &self,
        predicate: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
    ) {
        let _: () = msg_send![
            self,
            encodeStartIf: predicate,
            offset: offset,
            comparison: comparison,
            referenceValue: reference_value
        ];
    }

    unsafe fn encode_start_else(&self) {
        let _: () = msg_send![self, encodeStartElse];
    }

    unsafe fn encode_end_if(&self) -> bool {
        let result: bool = msg_send![self, encodeEndIf];
        result
    }
}

impl<T> ComputeEncoderConditional for T
where
    T: ComputeEncoderRawConditional,
{
    fn condition<IfBlock, ElseBlock>(
        &self,
        predicate: Option<&ProtocolObject<dyn MTLBuffer>>,
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
