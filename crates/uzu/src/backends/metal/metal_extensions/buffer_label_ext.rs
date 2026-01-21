use metal::{MTLBuffer, MTLComputeCommandEncoder, MTLTexture};
use objc2::{msg_send, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;

/// Extension trait to provide label() method for ProtocolObject<dyn MTLBuffer>
pub trait BufferLabelExt {
    fn label(&self) -> Option<String>;
    fn set_label(&self, label: Option<&str>);
}

impl BufferLabelExt for ProtocolObject<dyn MTLBuffer> {
    fn label(&self) -> Option<String> {
        let label: Option<Retained<NSString>> = unsafe { msg_send![self, label] };
        label.map(|label| label.to_string())
    }

    fn set_label(&self, label: Option<&str>) {
        unsafe {
            let _: () = msg_send![
                self,
                setLabel: label.map(NSString::from_str).as_deref()
            ];
        }
    }
}

impl BufferLabelExt for ProtocolObject<dyn MTLTexture> {
    fn label(&self) -> Option<String> {
        let label: Option<Retained<NSString>> = unsafe { msg_send![self, label] };
        label.map(|label| label.to_string())
    }

    fn set_label(&self, label: Option<&str>) {
        unsafe {
            let _: () = msg_send![
                self,
                setLabel: label.map(NSString::from_str).as_deref()
            ];
        }
    }
}

impl BufferLabelExt for ProtocolObject<dyn MTLComputeCommandEncoder> {
    fn label(&self) -> Option<String> {
        let label: Option<Retained<NSString>> = unsafe { msg_send![self, label] };
        label.map(|label| label.to_string())
    }

    fn set_label(&self, label: Option<&str>) {
        unsafe {
            let _: () = msg_send![
                self,
                setLabel: label.map(NSString::from_str).as_deref()
            ];
        }
    }
}
