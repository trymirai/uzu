use objc2::msg_send;

use crate::backends::metal::{
    MTLBuffer, MTLCompareFunction, MTLComputeCommandEncoder, ProtocolObject,
};

/// Low-level, unsafe conditional control of Metal encoders.
/// This is internal; users should prefer the safe `ComputeEncoderConditional::condition`.
trait ComputeEncoderRawConditional {
    fn encode_start_if(
        &self,
        predicate: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
    );

    fn encode_start_else(&self);

    fn encode_end_if(&self) -> bool;
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
    fn encode_start_if(
        &self,
        predicate: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        comparison: MTLCompareFunction,
        reference_value: u32,
    ) {
        let _: () = unsafe {
            msg_send![
                self,
                encodeStartIf: predicate,
                offset: offset,
                comparison: comparison,
                referenceValue: reference_value
            ]
        };
    }

    fn encode_start_else(&self) {
        let _: () = unsafe { msg_send![self, encodeStartElse] };
    }

    fn encode_end_if(&self) -> bool {
        let result: bool = unsafe { msg_send![self, encodeEndIf] };
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
            (Some(p), Some(else_fn)) => {
                self.encode_start_if(p, 0, MTLCompareFunction::Equal, 0);
                if_block();
                self.encode_start_else();
                else_fn();
                self.encode_end_if();
            },
            (Some(p), None) => {
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
