use metal::{MTL4ComputeCommandEncoder, MTLBuffer};
use obfstr::obfstr;
use objc2::{
    Message,
    runtime::{Bool, MessageReceiver, ProtocolObject},
};

use crate::backends::metal::metal_extensions::device_extensions::register_selector;

#[repr(u64)]
#[derive(Clone, Copy, Debug)]
pub enum PredicateComparison {
    Equal = 0,
    NotEqual = 1,
    Greater = 2,
    GreaterEqual = 3,
    Less = 4,
    LessEqual = 5,
}

pub trait ComputeCommandEncoderExt: MTL4ComputeCommandEncoder + Message + Sized {
    fn encode_start_if(
        &self,
        predicate: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        comparison: PredicateComparison,
        reference: u32,
    ) {
        let selector = register_selector(obfstr!("encodeStartIf:offset:comparison:referenceValue:"));
        let _: () = unsafe { self.send_message(selector, (predicate, offset, comparison as u64, reference)) };
    }

    fn encode_start_else(&self) {
        let _: () = unsafe { self.send_message(register_selector(obfstr!("encodeStartElse")), ()) };
    }

    fn encode_end_if(&self) -> bool {
        let ended: Bool = unsafe { self.send_message(register_selector(obfstr!("encodeEndIf")), ()) };
        ended.as_bool()
    }

    fn encode_start_while(
        &self,
        predicate: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        comparison: PredicateComparison,
        reference: u32,
    ) {
        let selector = register_selector(obfstr!("encodeStartWhile:offset:comparison:referenceValue:"));
        let _: () = unsafe { self.send_message(selector, (predicate, offset, comparison as u64, reference)) };
    }

    fn encode_end_while(&self) -> bool {
        let ended: Bool = unsafe { self.send_message(register_selector(obfstr!("encodeEndWhile")), ()) };
        ended.as_bool()
    }

    fn encode_start_do_while(&self) {
        let _: () = unsafe { self.send_message(register_selector(obfstr!("encodeStartDoWhile")), ()) };
    }

    fn encode_end_do_while(
        &self,
        predicate: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        comparison: PredicateComparison,
        reference: u32,
    ) -> bool {
        let selector = register_selector(obfstr!("encodeEndDoWhile:offset:comparison:referenceValue:"));
        let ended: Bool = unsafe { self.send_message(selector, (predicate, offset, comparison as u64, reference)) };
        ended.as_bool()
    }
}

impl ComputeCommandEncoderExt for ProtocolObject<dyn MTL4ComputeCommandEncoder> {}
