use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::prelude::*;

#[derive(Debug, Clone, Error)]
pub enum ResumeDataError {
    #[error("I/O error: {0}")]
    IoError(String),

    #[error("Failed to parse resume data: {0}")]
    ParseError(String),

    #[error("Decoded object is not a dictionary")]
    NotADictionary,

    #[error("Failed to serialize resume data")]
    SerializationFailed,
}

impl From<Retained<NSError>> for ResumeDataError {
    fn from(error: Retained<NSError>) -> Self {
        ResumeDataError::ParseError(error.localizedDescription().to_string())
    }
}

/// Parsed resume data from NSURLSessionDownloadTask
///
/// This struct allows reading resume data to extract download progress information.
#[derive(Debug, Clone)]
pub struct URLSessionDownloadTaskResumeData {
    /// Number of bytes received so far
    pub bytes_received: Option<u64>,
    /// Expected total bytes to receive
    pub bytes_expected_to_receive: Option<u64>,
    /// Local path where resume data was originally stored (iOS < 10.2)
    pub local_path: Option<PathBuf>,
    /// Temporary file name in the system temp directory
    pub temp_file_name: Option<String>,
    /// Byte range for resume (as raw object, needs further parsing if needed)
    pub byte_range: Option<String>,
    /// Original request data (NSKeyedArchive encoded NSURLRequest)
    pub original_request: Option<Vec<u8>>,
    /// Current request data (NSKeyedArchive encoded NSURLRequest)
    pub current_request: Option<Vec<u8>>,
    /// Resume data version
    pub version: Option<u64>,
}

impl URLSessionDownloadTaskResumeData {
    /// Parse resume data from a file path
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ResumeDataError> {
        let data = std::fs::read(path.as_ref()).map_err(|e| ResumeDataError::IoError(e.to_string()))?;
        Self::from_bytes(&data)
    }

    /// Parse resume data from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ResumeDataError> {
        let nsdata = NSData::with_bytes(bytes);
        let dict = Self::parse_resume_dict(&nsdata)?;

        Ok(Self {
            bytes_received: Self::get_number_value(&dict, "NSURLSessionResumeBytesReceived"),
            bytes_expected_to_receive: Self::get_number_value(&dict, "NSURLSessionResumeBytesExpectedToReceive"),
            local_path: Self::get_string_value(&dict, "NSURLSessionResumeInfoLocalPath").map(PathBuf::from),
            temp_file_name: Self::get_string_value(&dict, "NSURLSessionResumeInfoTempFileName"),
            byte_range: Self::get_string_value(&dict, "NSURLSessionResumeByteRange"),
            original_request: Self::get_data_value(&dict, "NSURLSessionResumeOriginalRequest"),
            current_request: Self::get_data_value(&dict, "NSURLSessionResumeCurrentRequest"),
            version: Self::get_number_value(&dict, "NSURLSessionResumeInfoVersion"),
        })
    }

    /// Serialize resume data back to bytes (standalone)
    ///
    /// Builds a minimal dictionary from the fields present on this struct
    /// and serializes it in binary PropertyList format. This does not depend
    /// on original resume data bytes.
    pub fn to_bytes(&self) -> Result<Box<[u8]>, ResumeDataError> {
        unsafe {
            // Create an empty NSDictionary<NSString, NSObject> to seed correct generics
            let empty: Retained<NSDictionary<NSString, NSObject>> = NSDictionary::new();
            let mutable_dict = NSMutableDictionary::dictionaryWithDictionary(&empty);

            if let Some(bytes_received) = self.bytes_received {
                mutable_dict.setObject_forKey(
                    &NSNumber::numberWithUnsignedLongLong(bytes_received),
                    &ProtocolObject::from_retained(NSString::from_str("NSURLSessionResumeBytesReceived")),
                );
            }

            if let Some(bytes_expected) = self.bytes_expected_to_receive {
                mutable_dict.setObject_forKey(
                    &NSNumber::numberWithUnsignedLongLong(bytes_expected),
                    &ProtocolObject::from_retained(NSString::from_str("NSURLSessionResumeBytesExpectedToReceive")),
                );
            }

            if let Some(ref local_path) = self.local_path {
                mutable_dict.setObject_forKey(
                    &NSString::from_str(&local_path.to_string_lossy()),
                    &ProtocolObject::from_retained(NSString::from_str("NSURLSessionResumeInfoLocalPath")),
                );
            }

            if let Some(ref temp_name) = self.temp_file_name {
                mutable_dict.setObject_forKey(
                    &NSString::from_str(temp_name),
                    &ProtocolObject::from_retained(NSString::from_str("NSURLSessionResumeInfoTempFileName")),
                );
            }

            if let Some(ref byte_range) = self.byte_range {
                mutable_dict.setObject_forKey(
                    &NSString::from_str(byte_range),
                    &ProtocolObject::from_retained(NSString::from_str("NSURLSessionResumeByteRange")),
                );
            }

            if let Some(ref original_req) = self.original_request {
                let nsdata = NSData::with_bytes(original_req);
                mutable_dict.setObject_forKey(
                    &nsdata,
                    &ProtocolObject::from_retained(NSString::from_str("NSURLSessionResumeOriginalRequest")),
                );
            }

            if let Some(ref current_req) = self.current_request {
                let nsdata = NSData::with_bytes(current_req);
                mutable_dict.setObject_forKey(
                    &nsdata,
                    &ProtocolObject::from_retained(NSString::from_str("NSURLSessionResumeCurrentRequest")),
                );
            }

            if let Some(ver) = self.version {
                mutable_dict.setObject_forKey(
                    &NSNumber::numberWithUnsignedLongLong(ver),
                    &ProtocolObject::from_retained(NSString::from_str("NSURLSessionResumeInfoVersion")),
                );
            }

            // Serialize to binary PropertyList format
            // NSURLSession accepts PropertyList format for resume data
            let data = NSPropertyListSerialization::dataWithPropertyList_format_options_error(
                &mutable_dict,
                NSPropertyListFormat::BinaryFormat_v1_0,
                0,
            )
            .ok()
            .ok_or_else(|| ResumeDataError::SerializationFailed)?;

            Ok(data.to_vec().into_boxed_slice())
        }
    }

    /// Save modified resume data to a file
    pub fn save_to_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(), ResumeDataError> {
        let bytes = self.to_bytes()?;
        std::fs::write(path.as_ref(), bytes).map_err(|e| ResumeDataError::IoError(e.to_string()))?;
        Ok(())
    }

    /// Get the temporary file path based on version
    pub fn temp_file_path(&self) -> Option<PathBuf> {
        if let Some(version) = self.version {
            if version > 1 {
                // iOS 10.2+ uses temp file name
                self.temp_file_name.as_ref().map(|name| std::env::temp_dir().join(name))
            } else {
                // iOS < 10.2 uses local path
                self.local_path.as_ref().map(|path| {
                    if let Some(name) = path.file_name() {
                        std::env::temp_dir().join(name)
                    } else {
                        path.clone()
                    }
                })
            }
        } else {
            // If version is missing, try temp_file_name first
            self.temp_file_name.as_ref().map(|name| std::env::temp_dir().join(name)).or_else(|| {
                self.local_path.as_ref().map(|path| {
                    if let Some(name) = path.file_name() {
                        std::env::temp_dir().join(name)
                    } else {
                        path.clone()
                    }
                })
            })
        }
    }

    /// Parse resume data dictionary using NSKeyedUnarchiver
    ///
    /// In beta versions and newer iOS, resumeData is NSKeyedArchive encoded
    /// instead of plain plist. Falls back to PropertyListSerialization for
    /// older formats.
    fn parse_resume_dict(data: &NSData) -> Result<Retained<NSDictionary<NSString, NSObject>>, ResumeDataError> {
        unsafe {
            let dict_cls = objc2::class!(NSDictionary);

            // Try NSKeyedUnarchiver first (NSKeyedArchive format) using modern API
            let allocated: Allocated<NSKeyedUnarchiver> = msg_send![NSKeyedUnarchiver::class(), alloc];

            match NSKeyedUnarchiver::initForReadingFromData_error(allocated, data) {
                Ok(unarchiver) => {
                    // Disable secure coding for NSURLSession resume data compatibility
                    unarchiver.setRequiresSecureCoding(false);

                    // Decode using the standard root object key
                    let key = NSString::from_str("NSKeyedArchiveRootObjectKey");

                    if let Some(obj) = unarchiver.decodeObjectForKey(&key) {
                        unarchiver.finishDecoding();
                        let is_dict: bool = msg_send![&obj, isKindOfClass: dict_cls];
                        if is_dict {
                            let dict = Retained::cast_unchecked::<NSDictionary<NSString, NSObject>>(obj);
                            return Ok(dict);
                        }
                    }

                    unarchiver.finishDecoding();
                },
                Err(_) => {},
            }

            // Fallback to PropertyListSerialization for legacy formats
            let obj = NSPropertyListSerialization::propertyListWithData_options_format_error(
                data,
                NSPropertyListMutabilityOptions::MutableContainersAndLeaves,
                core::ptr::null_mut::<NSPropertyListFormat>(),
            )
            .ok()
            .ok_or_else(|| {
                ResumeDataError::ParseError("Both NSKeyedUnarchiver and PropertyListSerialization failed".to_string())
            })?;

            // Check if it's a dictionary
            let is_dict: bool = msg_send![&obj, isKindOfClass: dict_cls];

            if is_dict {
                Ok(Retained::cast_unchecked(obj))
            } else {
                Err(ResumeDataError::NotADictionary)
            }
        }
    }

    /// Get a u64 value from the dictionary
    fn get_number_value(
        dict: &NSDictionary<NSString, NSObject>,
        key: &str,
    ) -> Option<u64> {
        let key_ns = NSString::from_str(key);
        dict.objectForKey(&key_ns).and_then(|obj| {
            let number_class = objc2::class!(NSNumber);
            let is_number = obj.isKindOfClass(number_class);

            if is_number {
                let num = unsafe { Retained::cast_unchecked::<NSNumber>(obj) };
                Some(num.unsignedLongLongValue())
            } else {
                None
            }
        })
    }

    /// Get a string value from the dictionary
    fn get_string_value(
        dict: &NSDictionary<NSString, NSObject>,
        key: &str,
    ) -> Option<String> {
        let key_ns = NSString::from_str(key);
        dict.objectForKey(&key_ns).and_then(|obj| {
            let string_class = objc2::class!(NSString);
            let is_string = obj.isKindOfClass(string_class);

            if is_string {
                let s = unsafe { Retained::cast_unchecked::<NSString>(obj) };
                Some(s.to_string())
            } else {
                None
            }
        })
    }

    /// Get data bytes from the dictionary
    fn get_data_value(
        dict: &NSDictionary<NSString, NSObject>,
        key: &str,
    ) -> Option<Vec<u8>> {
        let key_ns = NSString::from_str(key);
        dict.objectForKey(&key_ns).and_then(|obj| {
            let data_class = objc2::class!(NSData);
            let is_data = obj.isKindOfClass(data_class);

            if is_data {
                let data = unsafe { Retained::cast_unchecked::<NSData>(obj) };
                Some(data.to_vec())
            } else {
                None
            }
        })
    }
}
