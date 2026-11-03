//! Small helpers for messaging the private framework's Foundation objects, whose kinds are
//! checked before they are messaged (the framework's layout is not a stable interface).

use std::panic::AssertUnwindSafe;

use objc2::{ClassType, msg_send, rc::Retained, runtime::AnyObject};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString};

use super::AneError;

/// Runs `body`, converting a raised Objective-C exception into an [`AneError`].
pub fn guarded<R>(
    step: &'static str,
    body: impl FnOnce() -> Result<R, AneError>,
) -> Result<R, AneError> {
    match objc2::exception::catch(AssertUnwindSafe(body)) {
        Ok(result) => result,
        Err(exception) => Err(AneError::Call {
            step,
            message: exception.map_or_else(|| "unknown Objective-C exception".to_string(), |e| e.to_string()),
        }),
    }
}

pub fn call_error(
    step: &'static str,
    error: Option<&NSError>,
) -> AneError {
    AneError::Call {
        step,
        message: error.map_or_else(|| "no error given".to_string(), |error| error.to_string()),
    }
}

pub fn is_kind_of<T: ClassType>(object: &AnyObject) -> bool {
    unsafe { msg_send![object, isKindOfClass: T::class()] }
}

pub fn object_for_key(
    dictionary: &AnyObject,
    key: &str,
) -> Option<Retained<AnyObject>> {
    if !is_kind_of::<NSDictionary>(dictionary) {
        return None;
    }
    let key = NSString::from_str(key);
    unsafe { msg_send![dictionary, objectForKey: &*key] }
}

pub fn array_items(array: &AnyObject) -> Option<Vec<Retained<AnyObject>>> {
    if !is_kind_of::<NSArray>(array) {
        return None;
    }
    let count: usize = unsafe { msg_send![array, count] };
    let items = (0..count)
        .map(|index| {
            let item: Retained<AnyObject> = unsafe { msg_send![array, objectAtIndex: index] };
            item
        })
        .collect();
    Some(items)
}

pub fn dictionary_keys(dictionary: &AnyObject) -> Option<Vec<Retained<AnyObject>>> {
    if !is_kind_of::<NSDictionary>(dictionary) {
        return None;
    }
    let keys: Retained<AnyObject> = unsafe { msg_send![dictionary, allKeys] };
    array_items(&keys)
}

pub fn unsigned(object: &AnyObject) -> Option<u64> {
    if !is_kind_of::<NSNumber>(object) {
        return None;
    }
    let value: u64 = unsafe { msg_send![object, unsignedLongLongValue] };
    Some(value)
}

pub fn string(object: &AnyObject) -> Option<String> {
    if !is_kind_of::<NSString>(object) {
        return None;
    }
    let string = unsafe { &*(object as *const AnyObject).cast::<NSString>() };
    Some(string.to_string())
}

pub fn number(value: u64) -> Retained<NSNumber> {
    NSNumber::new_u64(value)
}
