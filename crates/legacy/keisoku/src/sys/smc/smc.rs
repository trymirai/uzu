use core::ffi::c_void;
#[cfg(feature = "hardware-control")]
use std::{
    thread,
    time::{Duration, Instant},
};

use obfstr::obfstr;
use objc2_core_foundation::{CFDictionary, CFRetained};
use objc2_io_kit::{
    IOConnectCallStructMethod, IOIteratorNext, IOObjectRelease, IOServiceClose, IOServiceGetMatchingServices,
    IOServiceMatching, IOServiceOpen, io_connect_t, io_iterator_t,
};

use super::{SmcError, SmcKeyData, SmcKeyInfo, fourcc};
use crate::{
    metrics::{Fan, FanMetrics},
    units::Rpm,
};

const KERNEL_INDEX_SMC: u32 = 2;
const SMC_CMD_READ_BYTES: u8 = 5;
const SMC_CMD_READ_KEYINFO: u8 = 9;

pub struct Smc {
    connection: io_connect_t,
}

impl Smc {
    #[allow(deprecated)]
    pub fn new() -> Option<Self> {
        Self::connect().ok()
    }

    #[allow(deprecated)]
    pub fn connect() -> Result<Self, SmcError> {
        let matching =
            unsafe { IOServiceMatching(obfstr::obfcstr!(c"AppleSMC").as_ptr()) }.ok_or(SmcError::Unavailable)?;
        let matching: CFRetained<CFDictionary> = unsafe { CFRetained::from_raw(CFRetained::into_raw(matching).cast()) };
        let mut iterator: io_iterator_t = 0;
        let result = unsafe { IOServiceGetMatchingServices(0, Some(matching), &mut iterator) };
        if result != 0 {
            return Err(SmcError::IoKit(result));
        }
        let device = IOIteratorNext(iterator);
        IOObjectRelease(iterator);
        if device == 0 {
            return Err(SmcError::Unavailable);
        }
        let mut connection: io_connect_t = 0;
        let result = unsafe { IOServiceOpen(device, libc::mach_task_self(), 0, &mut connection) };
        IOObjectRelease(device);
        if result != 0 {
            return Err(SmcError::IoKit(result));
        }
        Ok(Self {
            connection,
        })
    }

    pub fn fans(&self) -> FanMetrics {
        let count = self.read_u8(obfstr!("FNum")).unwrap_or(0);
        let fans = (0..count)
            .map(|index| Fan {
                actual: Rpm(self.fan_speed(index, obfstr!("Ac"))),
                minimum: Rpm(self.fan_speed(index, obfstr!("Mn"))),
                maximum: Rpm(self.fan_speed(index, obfstr!("Mx"))),
                target: Rpm(self.fan_speed(index, obfstr!("Tg"))),
            })
            .collect();
        FanMetrics {
            fans,
        }
    }

    fn fan_speed(
        &self,
        index: u8,
        suffix: &str,
    ) -> f32 {
        self.read_f32(&format!("{}{}{}", obfstr!("F"), index, suffix)).unwrap_or(0.0)
    }

    fn read_f32(
        &self,
        key: &str,
    ) -> Option<f32> {
        self.read_key(key).ok()?.as_f32().ok()
    }

    fn read_u8(
        &self,
        key: &str,
    ) -> Option<u8> {
        self.read_key(key).ok()?.as_u8().ok()
    }

    pub fn read_key(
        &self,
        key: &str,
    ) -> Result<SmcKeyData, SmcError> {
        self.read(fourcc(key).ok_or(SmcError::InvalidData {
            key: 0,
            reason: "key must contain four bytes",
        })?)
    }

    fn read(
        &self,
        key: u32,
    ) -> Result<SmcKeyData, SmcError> {
        let info = self.call(&SmcKeyData {
            key,
            data8: SMC_CMD_READ_KEYINFO,
            ..Default::default()
        })?;
        if info.key_info.data_size == 0 || info.key_info.data_size > 32 {
            return Err(SmcError::InvalidData {
                key,
                reason: "payload size must be between 1 and 32",
            });
        }
        let mut value = self.call(&SmcKeyData {
            key,
            key_info: SmcKeyInfo {
                data_size: info.key_info.data_size,
                ..Default::default()
            },
            data8: SMC_CMD_READ_BYTES,
            ..Default::default()
        })?;
        value.key = key;
        value.key_info = info.key_info;
        Ok(value)
    }

    #[cfg(feature = "hardware-control")]
    pub fn write_checked(
        &self,
        original: &SmcKeyData,
        bytes: &[u8],
    ) -> Result<(), SmcError> {
        if bytes.is_empty() || bytes.len() > 32 || bytes.len() != original.key_info.data_size as usize {
            return Err(SmcError::InvalidData {
                key: original.key,
                reason: "write payload size mismatch",
            });
        }
        let mut input = SmcKeyData {
            key: original.key,
            key_info: original.key_info,
            data8: 6,
            ..Default::default()
        };
        input.bytes[..bytes.len()].copy_from_slice(bytes);
        self.call(&input)?;
        verify_readback(original, bytes, || self.read(original.key))
    }

    fn call(
        &self,
        input: &SmcKeyData,
    ) -> Result<SmcKeyData, SmcError> {
        let mut output = SmcKeyData::default();
        let mut output_size = core::mem::size_of::<SmcKeyData>();
        let result = unsafe {
            IOConnectCallStructMethod(
                self.connection,
                KERNEL_INDEX_SMC,
                (input as *const SmcKeyData).cast::<c_void>(),
                core::mem::size_of::<SmcKeyData>(),
                (&mut output as *mut SmcKeyData).cast::<c_void>(),
                &mut output_size,
            )
        };
        if result != 0 {
            return Err(SmcError::IoKit(result));
        }
        if output_size != core::mem::size_of::<SmcKeyData>() {
            return Err(SmcError::InvalidData {
                key: input.key,
                reason: "unexpected reply size",
            });
        }
        match output.result {
            0 => Ok(output),
            0x84 => Err(SmcError::MissingKey(input.key)),
            result => Err(SmcError::Firmware {
                key: input.key,
                result,
            }),
        }
    }
}

impl Drop for Smc {
    fn drop(&mut self) {
        IOServiceClose(self.connection);
    }
}

#[cfg(feature = "hardware-control")]
fn verify_readback(
    original: &SmcKeyData,
    bytes: &[u8],
    mut read: impl FnMut() -> Result<SmcKeyData, SmcError>,
) -> Result<(), SmcError> {
    // Fan target registers can lag an accepted write by about a second.
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let actual = read()?;
        if actual.key_info.data_size != original.key_info.data_size
            || actual.key_info.data_type != original.key_info.data_type
        {
            return Err(SmcError::ReadbackMismatch(original.key));
        }
        if actual.bytes[..bytes.len()] == *bytes {
            return Ok(());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(SmcError::ReadbackMismatch(original.key));
        }
        thread::sleep(remaining.min(Duration::from_millis(100)));
    }
}

#[cfg(all(test, feature = "hardware-control"))]
mod tests {
    use super::*;

    #[test]
    fn readback_waits_for_fan_target_to_settle() {
        let target = SmcKeyData {
            key_info: SmcKeyInfo {
                data_size: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let bytes = 7826.0_f32.to_le_bytes();
        let mut reads = 0;
        verify_readback(&target, &bytes, || {
            reads += 1;
            let mut actual = target;
            if reads == 3 {
                actual.bytes[..4].copy_from_slice(&bytes);
            }
            Ok(actual)
        })
        .unwrap();
        assert_eq!(reads, 3);
    }

    #[test]
    fn readback_times_out_when_target_never_settles() {
        let target = SmcKeyData::default();
        let mut reads = 0;
        assert!(matches!(
            verify_readback(&target, &[1], || {
                reads += 1;
                Ok(target)
            }),
            Err(SmcError::ReadbackMismatch(_))
        ));
        assert!(reads > 1);
    }

    #[test]
    fn readback_rejects_changed_metadata_and_io_errors() {
        let original = SmcKeyData::default();
        for change_type in [false, true] {
            let mut reads = 0;
            assert!(matches!(
                verify_readback(&original, &[0], || {
                    reads += 1;
                    let mut actual = original;
                    if change_type {
                        actual.key_info.data_type = 1;
                    } else {
                        actual.key_info.data_size = 1;
                    }
                    Ok(actual)
                }),
                Err(SmcError::ReadbackMismatch(_))
            ));
            assert_eq!(reads, 1);
        }
        assert!(matches!(verify_readback(&original, &[0], || Err(SmcError::IoKit(-1))), Err(SmcError::IoKit(-1))));
    }
}
