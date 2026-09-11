use objc2::{AnyThread, rc::Retained, runtime::AnyObject};
use objc2_foundation::{
    NSData, NSDictionary, NSKeyedUnarchiver, NSNumber, NSPropertyListFormat, NSPropertyListMutabilityOptions,
    NSPropertyListSerialization, NSString,
};

pub struct ResumeData(Vec<u8>);

impl ResumeData {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    pub fn bytes_received(&self) -> Option<u64> {
        if self.0.is_empty() {
            return None;
        }
        let data = NSData::with_bytes(&self.0);
        let dictionary =
            Self::keyed_archive(&data).or_else(|| Self::property_list(&data))?.downcast::<NSDictionary>().ok()?;
        let value = dictionary.objectForKey(&NSString::from_str("NSURLSessionResumeBytesReceived"))?;
        Some(value.downcast::<NSNumber>().ok()?.unsignedLongLongValue())
    }

    fn keyed_archive(data: &NSData) -> Option<Retained<AnyObject>> {
        unsafe {
            let unarchiver = NSKeyedUnarchiver::initForReadingFromData_error(NSKeyedUnarchiver::alloc(), data).ok()?;
            unarchiver.setRequiresSecureCoding(false);
            let object = unarchiver.decodeObjectForKey(&NSString::from_str("NSKeyedArchiveRootObjectKey"));
            unarchiver.finishDecoding();
            object
        }
    }

    fn property_list(data: &NSData) -> Option<Retained<AnyObject>> {
        unsafe {
            NSPropertyListSerialization::propertyListWithData_options_format_error(
                data,
                NSPropertyListMutabilityOptions::MutableContainersAndLeaves,
                std::ptr::null_mut::<NSPropertyListFormat>(),
            )
            .ok()
        }
    }
}
