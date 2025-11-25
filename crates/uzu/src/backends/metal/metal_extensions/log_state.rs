use std::sync::OnceLock;

use metal::{
    CommandQueue, Device,
    foreign_types::{ForeignType, ForeignTypeRef},
};
use mpsgraph::CommandBuffer as MPSCommandBuffer;
use objc2::{msg_send, runtime::AnyObject};

static GLOBAL_LOG_STATE: OnceLock<Option<GlobalLogState>> = OnceLock::new();
static BIND_LOG_STATE_AVAILABLE: OnceLock<bool> = OnceLock::new();
static SET_PRIVATE_LOGGING_BUFFER_AVAILABLE: OnceLock<bool> = OnceLock::new();
static SET_LOGS_AVAILABLE: OnceLock<bool> = OnceLock::new();

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
        std::env::var("UZU_METAL_LOGGING").ok().and_then(|s| {
            match s.to_lowercase().as_str() {
                "debug" => Some(LogLevel::Debug),
                "info" => Some(LogLevel::Info),
                "notice" => Some(LogLevel::Notice),
                "error" => Some(LogLevel::Error),
                "fault" => Some(LogLevel::Fault),
                _ => None,
            }
        })
    }
}

pub fn initialize_metal_logging(device: &Device) -> Option<isize> {
    let level = LogLevel::from_env()?;
    let mtl_level = level.to_mtl_level();

    GLOBAL_LOG_STATE.get_or_init(|| unsafe {
        // Create MTLLogStateDescriptor
        let descriptor_class: *mut AnyObject =
            msg_send![objc2::class!(MTLLogStateDescriptor), alloc];
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

        // Release descriptor (it was created with alloc/init so we own it)
        let _: () = msg_send![descriptor, release];

        if log_state_ptr.is_null() {
            if !error.is_null() {
                let err_desc: *mut AnyObject =
                    msg_send![error, localizedDescription];
                let err_str: *const i8 = msg_send![err_desc, UTF8String];
                let err_string =
                    std::ffi::CStr::from_ptr(err_str).to_string_lossy();
                eprintln!("[UZU] Failed to create log state: {}", err_string);
            }
            None
        } else {
            // Add log handler to print to stderr
            add_log_handler(log_state_ptr);

            eprintln!("[UZU] Metal logging enabled at level: {:?}", level);
            Some(GlobalLogState {
                log_state_ptr,
                level: mtl_level,
            })
        }
    });

    Some(mtl_level)
}

#[cfg(target_os = "macos")]
unsafe fn add_log_handler(log_state_ptr: *mut AnyObject) {
    use block2::RcBlock;

    // Create a block that prints logs to stderr
    let handler = RcBlock::new(
        |subsystem: *mut AnyObject,
         category: *mut AnyObject,
         log_level: isize,
         message: *mut AnyObject| {
            let level_str = match log_level {
                1 => "DEBUG",
                2 => "INFO",
                3 => "NOTICE",
                4 => "ERROR",
                5 => "FAULT",
                _ => "UNKNOWN",
            };

            // Extract message string
            let msg_str: *const i8 = msg_send![message, UTF8String];
            if !msg_str.is_null() {
                let message_cstr = unsafe { std::ffi::CStr::from_ptr(msg_str) };
                if let Ok(message_str) = message_cstr.to_str() {
                    eprintln!("[Metal {}] {}", level_str, message_str);
                }
            }
        },
    );

    // Add the handler to the log state
    let handler_ptr = &*handler as *const _ as *mut AnyObject;
    let _: () = msg_send![log_state_ptr, addLogHandler: handler_ptr];

    // Keep the block alive
    std::mem::forget(handler);
}

#[cfg(not(target_os = "macos"))]
unsafe fn add_log_handler(_log_state_ptr: *mut AnyObject) {
    // iOS/tvOS/watchOS - logs go to system log by default
    eprintln!("[UZU] Metal logs will appear in system console");
}

pub trait CommandQueueLoggingExt {
    fn new_command_buffer_with_logging(&self) -> metal::CommandBuffer;
}

impl CommandQueueLoggingExt for CommandQueue {
    fn new_command_buffer_with_logging(&self) -> metal::CommandBuffer {
        // Note: bindLogState: exists in private Metal headers but is not public API
        // We must use MTLCommandBufferDescriptor at creation time
        if let Some(global_state) =
            GLOBAL_LOG_STATE.get().and_then(|s| s.as_ref())
        {
            unsafe {
                // Create MTLCommandBufferDescriptor
                let descriptor_class: *mut AnyObject =
                    msg_send![objc2::class!(MTLCommandBufferDescriptor), alloc];
                let descriptor: *mut AnyObject =
                    msg_send![descriptor_class, init];

                // Set log state
                let _: () = msg_send![
                    descriptor,
                    setLogState: global_state.log_state_ptr
                ];

                let queue_ptr = self.as_ptr() as *mut AnyObject;

                // Create command buffer with descriptor
                let cmd_buffer_ptr: *mut AnyObject = msg_send![
                    queue_ptr,
                    commandBufferWithDescriptor: descriptor
                ];

                // Release descriptor
                let _: () = msg_send![descriptor, release];

                if !cmd_buffer_ptr.is_null() {
                    let _: *mut AnyObject = msg_send![cmd_buffer_ptr, retain];

                    // Attempt to bind log state on raw Metal command buffers (private, no-arg)
                    let _ = std::panic::catch_unwind(
                        std::panic::AssertUnwindSafe(|| {
                            let _: () = msg_send![cmd_buffer_ptr, bindLogState];
                        }),
                    );

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

/// Create an MPSCommandBuffer with logging support
///
/// This attempts to call private logging methods on the underlying Metal command buffer
/// obtained from MPSCommandBuffer. We probe for three private API methods in order:
/// 1. `bindLogState:` - binds an MTLLogState
/// 2. `setPrivateLoggingBuffer:` - sets a private logging buffer
/// 3. `setLogs:` - sets logs directly
///
/// See: https://developer.limneos.net/index.php?ios=18.1&framework=Metal.framework&header=_MTLCommandBuffer.h
pub fn new_mps_command_buffer_with_logging(
    command_queue: &CommandQueue
) -> objc2::rc::Retained<MPSCommandBuffer> {
    let mps_cmd_buffer = MPSCommandBuffer::from_command_queue(command_queue);

    // Try to bind log state to the underlying Metal command buffer
    if let Some(global_state) = GLOBAL_LOG_STATE.get().and_then(|s| s.as_ref())
    {
        unsafe {
            let root_command_buffer = mps_cmd_buffer.root_command_buffer();
            let root_cb_ptr = root_command_buffer.as_ptr() as *mut AnyObject;

            // Probe bindLogState: (cached check)
            let bind_log_state_available = BIND_LOG_STATE_AVAILABLE
                .get_or_init(|| {
                    let result = std::panic::catch_unwind(
                        std::panic::AssertUnwindSafe(|| {
                            let _: () = msg_send![root_cb_ptr, bindLogState];
                        }),
                    );
                    result.is_ok()
                });

            // Probe setPrivateLoggingBuffer: (cached check)
            let set_private_logging_buffer_available = SET_PRIVATE_LOGGING_BUFFER_AVAILABLE.get_or_init(|| {
                let result = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| {
                        let _: () = msg_send![root_cb_ptr, setPrivateLoggingBuffer: global_state.log_state_ptr];
                    }),
                );
                result.is_ok()
            });

            // Probe setLogs: (cached check)
            let set_logs_available = SET_LOGS_AVAILABLE.get_or_init(|| {
                let result = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| {
                        let _: () = msg_send![root_cb_ptr, setLogs: global_state.log_state_ptr];
                    }),
                );
                result.is_ok()
            });

            // Report availability status (only once)
            static REPORTED: OnceLock<()> = OnceLock::new();
            REPORTED.get_or_init(|| {
                if *bind_log_state_available {
                    eprintln!("[UZU] bindLogState: available - shader logging may work!");
                } else if *set_private_logging_buffer_available {
                    eprintln!("[UZU] setPrivateLoggingBuffer: available - shader logging may work!");
                } else if *set_logs_available {
                    eprintln!("[UZU] setLogs: available - shader logging may work!");
                } else {
                    eprintln!(
                        "[UZU] Warning: No private logging methods available (bindLogState:, setPrivateLoggingBuffer:, setLogs:). Shader logs will not appear."
                    );
                }
            });

            // Try methods in order of preference for subsequent command buffers
            if *bind_log_state_available {
                {
                    // Attach log state via descriptor if available, then bind (private)
                    let descriptor_class: *mut AnyObject = msg_send![
                        objc2::class!(MTLCommandBufferDescriptor),
                        alloc
                    ];
                    let descriptor: *mut AnyObject =
                        msg_send![descriptor_class, init];
                    let _: () = msg_send![descriptor, setLogState: global_state.log_state_ptr];

                    // Try private configWithCommandBufferDescriptor: (graceful if unavailable)
                    let _ = std::panic::catch_unwind(
                        std::panic::AssertUnwindSafe(|| {
                            let _: () = msg_send![
                                root_cb_ptr,
                                configWithCommandBufferDescriptor: descriptor
                            ];
                        }),
                    );

                    // Release descriptor regardless
                    let _: () = msg_send![descriptor, release];

                    // Finally bind log state (no-arg) - required on some drivers
                    let _ = std::panic::catch_unwind(
                        std::panic::AssertUnwindSafe(|| {
                            let _: () = msg_send![root_cb_ptr, bindLogState];
                        }),
                    );
                }
            } else if *set_private_logging_buffer_available {
                let _: () = msg_send![root_cb_ptr, setPrivateLoggingBuffer: global_state.log_state_ptr];
            } else if *set_logs_available {
                let _: () =
                    msg_send![root_cb_ptr, setLogs: global_state.log_state_ptr];
            }
        }
    }

    mps_cmd_buffer
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
