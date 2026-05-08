use std::path::Path;

use objc2::{AnyThread, class, msg_send, rc::Retained, runtime::AnyObject};
use objc2_foundation::{
    NSData, NSDictionary, NSKeyedUnarchiver, NSNumber, NSObject, NSObjectProtocol, NSPropertyListFormat,
    NSPropertyListMutabilityOptions, NSPropertyListSerialization, NSString,
};

const RESUME_BYTES_RECEIVED_KEY: &str = "NSURLSessionResumeBytesReceived";

pub(crate) fn read_resume_progress(part_path: &Path) -> Option<u64> {
    let bytes = std::fs::read(part_path).ok()?;
    let nsdata = NSData::with_bytes(&bytes);
    let dict = parse_resume_dict(&nsdata)?;
    read_unsigned(&dict, RESUME_BYTES_RECEIVED_KEY)
}

fn parse_resume_dict(data: &NSData) -> Option<Retained<NSDictionary<NSString, NSObject>>> {
    // Newer OS releases use NSKeyedArchive; older ones use a plain plist.
    if let Some(dict) = parse_via_keyed_unarchiver(data) {
        return Some(dict);
    }
    parse_via_property_list(data)
}

fn parse_via_keyed_unarchiver(data: &NSData) -> Option<Retained<NSDictionary<NSString, NSObject>>> {
    unsafe {
        let allocated = NSKeyedUnarchiver::alloc();
        let unarchiver = NSKeyedUnarchiver::initForReadingFromData_error(allocated, data).ok()?;
        unarchiver.setRequiresSecureCoding(false);

        let key = NSString::from_str("NSKeyedArchiveRootObjectKey");
        let object = unarchiver.decodeObjectForKey(&key);
        unarchiver.finishDecoding();

        let object = object?;
        let dict_class = class!(NSDictionary);
        let raw: *mut AnyObject = Retained::into_raw(object).cast();
        let is_dict: bool = msg_send![raw, isKindOfClass: dict_class];
        if !is_dict {
            // Balance the `into_raw` by reclaiming ownership so the object drops.
            let _ = Retained::<AnyObject>::from_raw(raw);
            return None;
        }
        let typed_ptr = raw.cast::<NSDictionary<NSString, NSObject>>();
        Retained::from_raw(typed_ptr)
    }
}

fn parse_via_property_list(data: &NSData) -> Option<Retained<NSDictionary<NSString, NSObject>>> {
    unsafe {
        let object = NSPropertyListSerialization::propertyListWithData_options_format_error(
            data,
            NSPropertyListMutabilityOptions::MutableContainersAndLeaves,
            core::ptr::null_mut::<NSPropertyListFormat>(),
        )
        .ok()?;
        let dict_class = class!(NSDictionary);
        let raw: *mut AnyObject = Retained::into_raw(object).cast();
        let is_dict: bool = msg_send![raw, isKindOfClass: dict_class];
        if !is_dict {
            // Balance the `into_raw` by reclaiming ownership so the object drops.
            let _ = Retained::<AnyObject>::from_raw(raw);
            return None;
        }
        let typed_ptr = raw.cast::<NSDictionary<NSString, NSObject>>();
        Retained::from_raw(typed_ptr)
    }
}

fn read_unsigned(
    dict: &NSDictionary<NSString, NSObject>,
    key: &str,
) -> Option<u64> {
    let key_ns = NSString::from_str(key);
    let object = dict.objectForKey(&key_ns)?;
    let number_class = class!(NSNumber);
    if !object.isKindOfClass(number_class) {
        return None;
    }
    let number = unsafe { Retained::cast_unchecked::<NSNumber>(object) };
    Some(number.unsignedLongLongValue())
}
