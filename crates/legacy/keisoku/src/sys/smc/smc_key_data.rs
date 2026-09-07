use super::{SmcError, SmcKeyInfo, SmcLimitData, SmcVersion, fourcc};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SmcKeyData {
    pub key: u32,
    pub version: SmcVersion,
    pub limit: SmcLimitData,
    pub key_info: SmcKeyInfo,
    pub result: u8,
    pub status: u8,
    pub data8: u8,
    pub data32: u32,
    pub bytes: [u8; 32],
}

const _: () = assert!(core::mem::size_of::<SmcKeyData>() == 80);
const _: () = assert!(core::mem::offset_of!(SmcKeyData, key_info) == 28);
const _: () = assert!(core::mem::offset_of!(SmcKeyData, result) == 40);
const _: () = assert!(core::mem::offset_of!(SmcKeyData, data8) == 42);
const _: () = assert!(core::mem::offset_of!(SmcKeyData, data32) == 44);
const _: () = assert!(core::mem::offset_of!(SmcKeyData, bytes) == 48);

impl SmcKeyData {
    pub fn as_u8(&self) -> Result<u8, SmcError> {
        self.validate_type(obfstr::obfstr!("ui8 "), 1)?;
        Ok(self.bytes[0])
    }

    pub fn as_f32(&self) -> Result<f32, SmcError> {
        self.validate_type(obfstr::obfstr!("flt "), 4)?;
        Ok(f32::from_le_bytes([self.bytes[0], self.bytes[1], self.bytes[2], self.bytes[3]]))
    }

    fn validate_type(
        &self,
        kind: &str,
        size: u32,
    ) -> Result<(), SmcError> {
        if Some(self.key_info.data_type) != fourcc(kind) || self.key_info.data_size != size {
            return Err(SmcError::InvalidData {
                key: self.key,
                reason: "unsupported payload type or size",
            });
        }
        Ok(())
    }
}
