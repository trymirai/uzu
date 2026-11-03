//! The private `AppleNeuralEngine.framework` classes and client, resolved and checked once.

use std::{ffi::CStr, sync::Arc};

use objc2::{
    msg_send,
    rc::Retained,
    runtime::{AnyClass, AnyObject, Sel},
    sel,
};

use super::{AneError, objc::guarded};

const FRAMEWORK: &CStr = c"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";

/// `QOS_CLASS_USER_INTERACTIVE`: prefill is on the user's critical path.
pub const ANE_QOS: u32 = 0x21;

pub struct AneRuntime {
    pub(super) client: Retained<AnyObject>,
    pub(super) model_class: &'static AnyClass,
    pub(super) surface_class: &'static AnyClass,
    pub(super) request_class: &'static AnyClass,
    pub(super) wait_event_class: &'static AnyClass,
    pub(super) signal_event_class: &'static AnyClass,
    pub(super) shared_events_class: &'static AnyClass,
}

// SAFETY: `_ANEClient`'s shared connection is the framework's process-wide client, which Core ML
// messages from arbitrary threads; the classes are immutable runtime metadata.
unsafe impl Send for AneRuntime {}
unsafe impl Sync for AneRuntime {}

fn class(name: &'static CStr) -> Result<&'static AnyClass, AneError> {
    AnyClass::get(name)
        .ok_or_else(|| AneError::Unavailable(format!("AppleNeuralEngine lacks {}", name.to_string_lossy())))
}

fn require_class_method(
    class: &AnyClass,
    selector: Sel,
) -> Result<(), AneError> {
    if class.class_method(selector).is_none() {
        return Err(AneError::Unavailable(format!(
            "{} lacks +{}",
            class.name().to_string_lossy(),
            selector.name().to_string_lossy()
        )));
    }
    Ok(())
}

fn require_instance_method(
    class: &AnyClass,
    selector: Sel,
) -> Result<(), AneError> {
    if !class.responds_to(selector) {
        return Err(AneError::Unavailable(format!(
            "{} lacks -{}",
            class.name().to_string_lossy(),
            selector.name().to_string_lossy()
        )));
    }
    Ok(())
}

impl AneRuntime {
    /// Loads the framework and checks every class and selector the runtime messages.
    pub fn new() -> Result<Arc<Self>, AneError> {
        let handle = unsafe { libc::dlopen(FRAMEWORK.as_ptr(), libc::RTLD_NOW) };
        if handle.is_null() {
            return Err(AneError::Unavailable("cannot load AppleNeuralEngine.framework".to_string()));
        }

        let model_class = class(c"_ANEModel")?;
        let client_class = class(c"_ANEClient")?;
        let surface_class = class(c"_ANEIOSurfaceObject")?;
        let request_class = class(c"_ANERequest")?;
        let wait_event_class = class(c"_ANESharedWaitEvent")?;
        let signal_event_class = class(c"_ANESharedSignalEvent")?;
        let shared_events_class = class(c"_ANESharedEvents")?;

        require_class_method(model_class, sel!(modelAtURL:key:))?;
        require_instance_method(model_class, sel!(modelAttributes))?;
        require_class_method(client_class, sel!(sharedConnection))?;
        for selector in [
            sel!(compileModel:options:qos:error:),
            sel!(compiledModelExistsFor:),
            sel!(purgeCompiledModel:),
            sel!(loadModel:options:qos:error:),
            sel!(unloadModel:options:qos:error:),
            sel!(evaluateWithModel:options:request:qos:error:),
        ] {
            require_instance_method(client_class, selector)?;
        }
        require_class_method(surface_class, sel!(objectWithIOSurface:))?;
        require_class_method(
            request_class,
            sel!(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:),
        )?;
        require_instance_method(request_class, sel!(setCompletionHandler:))?;
        require_class_method(wait_event_class, sel!(waitEventWithValue:sharedEvent:eventType:))?;
        require_class_method(signal_event_class, sel!(signalEventWithValue:symbolIndex:eventType:sharedEvent:))?;
        require_class_method(shared_events_class, sel!(sharedEventsWithSignalEvents:waitEvents:))?;

        let client = guarded("client connection", || {
            let client: Option<Retained<AnyObject>> = unsafe { msg_send![client_class, sharedConnection] };
            client.ok_or_else(|| AneError::Unavailable("no _ANEClient shared connection".to_string()))
        })?;

        Ok(Arc::new(Self {
            client,
            model_class,
            surface_class,
            request_class,
            wait_event_class,
            signal_event_class,
            shared_events_class,
        }))
    }
}
