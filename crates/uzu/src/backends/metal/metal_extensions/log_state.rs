use metal::{CommandQueue, Device, foreign_types::ForeignType};
use objc2::{msg_send, runtime::AnyObject};
use std::sync::OnceLock;

static GLOBAL_LOG_STATE: OnceLock<Option<GlobalLogState>> = OnceLock::new();

struct GlobalLogState {
    log_state_ptr: *mut AnyObject,
    level: isize,
}

unsafe impl Send for GlobalLogState {}
unsafe impl Sync for GlobalLogState {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Notice,
    Error,
    Fault,
}

impl LogLevel {
    fn to_mtl_level(self) -> isize {
        match self {
            LogLevel::Debug => 1,
            LogLevel::Info => 2,
            LogLevel::Notice => 3,
            LogLevel::Error => 4,
            LogLevel::Fault => 5,
        }
    }

    pub fn from_env() -> Option<Self> {
        std::env::var("UZU_METAL_LOGGING")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "debug" => Some(LogLevel::Debug),
                "info" => Some(LogLevel::Info),
                "notice" => Some(LogLevel::Notice),
                "error" => Some(LogLevel::Error),
                "fault" => Some(LogLevel::Fault),
                _ => None,
            })
    }
}

pub fn initialize_metal_logging(device: &Device) -> Option<isize> {
    let level = LogLevel::from_env()?;
    let mtl_level = level.to_mtl_level();

    GLOBAL_LOG_STATE.get_or_init(|| unsafe {
        // Create MTLLogStateDescriptor
        let descriptor_class: *mut AnyObject = msg_send![
            objc2::class!(MTLLogStateDescriptor),
            alloc
        ];
        let descriptor: *mut AnyObject = msg_send![descriptor_class, init];
        
        // Set level
        let _: () = msg_send![descriptor, setLevel: mtl_level];
        
        // Set buffer size to 1MB
        let _: () = msg_send![descriptor, setBufferSize: 1024 * 1024isize];
        
        let device_ptr = device.as_ptr() as *mut AnyObject;
        
        // Call newLogStateWithDescriptor:error:
        let mut error: *mut AnyObject = std::ptr::null_mut();
        let log_state_ptr: *mut AnyObject = msg_send![
            device_ptr,
            newLogStateWithDescriptor: descriptor,
            error: &mut error
        ];

        // Retain the log state
        if !log_state_ptr.is_null() {
            let _: *mut AnyObject = msg_send![log_state_ptr, retain];
        }

        // Release descriptor
        let _: () = msg_send![descriptor, release];

        if log_state_ptr.is_null() {
            if !error.is_null() {
                let err_desc: *mut AnyObject = msg_send![error, localizedDescription];
                let err_str: *const i8 = msg_send![err_desc, UTF8String];
                let err_string = std::ffi::CStr::from_ptr(err_str).to_string_lossy();
                eprintln!("[UZU] Failed to create log state: {}", err_string);
            }
            None
        } else {
            eprintln!("[UZU] Metal logging enabled at level: {:?}", level);
            Some(GlobalLogState {
                log_state_ptr,
                level: mtl_level,
            })
        }
    });

    Some(mtl_level)
}

pub trait CommandQueueLoggingExt {
    fn new_command_buffer_with_logging(&self) -> metal::CommandBuffer;
}

impl CommandQueueLoggingExt for CommandQueue {
    fn new_command_buffer_with_logging(&self) -> metal::CommandBuffer {
        if let Some(global_state) = GLOBAL_LOG_STATE.get().and_then(|s| s.as_ref()) {
            unsafe {
                // Create MTLCommandBufferDescriptor
                let descriptor_class: *mut AnyObject = msg_send![
                    objc2::class!(MTLCommandBufferDescriptor),
                    alloc
                ];
                let descriptor: *mut AnyObject = msg_send![descriptor_class, init];
                
                // Set log state
                let _: () = msg_send![descriptor, setLogState: global_state.log_state_ptr];
                
                let queue_ptr = self.as_ptr() as *mut AnyObject;
                
                // Create command buffer with descriptor
                let cmd_buffer_ptr: *mut AnyObject = msg_send![
                    queue_ptr,
                    commandBufferWithDescriptor: descriptor
                ];
                
                // Release descriptor
                let _: () = msg_send![descriptor, release];

                if !cmd_buffer_ptr.is_null() {
                    metal::CommandBuffer::from_ptr(cmd_buffer_ptr as *mut _)
                } else {
                    // Fallback to regular command buffer
                    self.new_command_buffer().to_owned()
                }
            }
        } else {
            // No logging enabled, use regular command buffer
            self.new_command_buffer().to_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_from_env() {
        std::env::set_var("UZU_METAL_LOGGING", "debug");
        assert_eq!(LogLevel::from_env(), Some(LogLevel::Debug));

        std::env::set_var("UZU_METAL_LOGGING", "INFO");
        assert_eq!(LogLevel::from_env(), Some(LogLevel::Info));

        std::env::set_var("UZU_METAL_LOGGING", "invalid");
        assert_eq!(LogLevel::from_env(), None);

        std::env::remove_var("UZU_METAL_LOGGING");
        assert_eq!(LogLevel::from_env(), None);
    }
}
