use metal::{MTLBuffer, MTLCompareFunction, MTLComputeCommandEncoder};
use objc2::{msg_send, rc::Retained, runtime::ProtocolObject};

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
        predicate: &Retained<ProtocolObject<dyn MTLBuffer>>,
        offset: usize,
        if_block: IfBlock,
        else_block: Option<ElseBlock>,
    ) where
        IfBlock: FnOnce(),
        ElseBlock: FnOnce();
}

impl ComputeEncoderRawConditional for ProtocolObject<dyn MTLComputeCommandEncoder> {
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
        predicate: &Retained<ProtocolObject<dyn MTLBuffer>>,
        offset: usize,
        if_block: IfBlock,
        else_block: Option<ElseBlock>,
    ) where
        IfBlock: FnOnce(),
        ElseBlock: FnOnce(),
    {
        self.encode_start_if(predicate, offset, MTLCompareFunction::Equal, 0);
        if_block();
        if let Some(else_block) = else_block {
            self.encode_start_else();
            else_block();
        }
        self.encode_end_if();
    }
}
