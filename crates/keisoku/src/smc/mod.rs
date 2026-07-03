mod fourcc;
mod key_data;

use core::ffi::c_void;

use obfstr::obfstr;
use objc2_core_foundation::{CFDictionary, CFRetained};
use objc2_io_kit::{
    IOConnectCallStructMethod, IOIteratorNext, IOObjectRelease, IOServiceClose, IOServiceGetMatchingServices,
    IOServiceMatching, IOServiceOpen, io_connect_t, io_iterator_t,
};

use self::{
    fourcc::fourcc,
    key_data::{SmcKeyData, SmcKeyInfo},
};
use crate::{
    metrics::{Fan, FanMetrics},
    units::{Rpm, Watts},
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
        let name = std::ffi::CString::new(obfstr!("AppleSMC")).ok()?;
        let matching = unsafe { IOServiceMatching(name.as_ptr()) }?;
        let matching: CFRetained<CFDictionary> = unsafe { CFRetained::from_raw(CFRetained::into_raw(matching).cast()) };
        let mut iterator: io_iterator_t = 0;
        if unsafe { IOServiceGetMatchingServices(0, Some(matching), &mut iterator) } != 0 {
            return None;
        }
        let device = IOIteratorNext(iterator);
        IOObjectRelease(iterator);
        if device == 0 {
            return None;
        }
        let mut connection: io_connect_t = 0;
        let result = unsafe { IOServiceOpen(device, libc::mach_task_self(), 0, &mut connection) };
        IOObjectRelease(device);
        (result == 0).then_some(Self {
            connection,
        })
    }

    pub fn package_watts(&self) -> Option<Watts> {
        self.read_f32(obfstr!("PSTR")).map(Watts)
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
        let (data_type, bytes) = self.read_key(key)?;
        (data_type == fourcc(obfstr!("flt "))?).then(|| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u8(
        &self,
        key: &str,
    ) -> Option<u8> {
        self.read_key(key).map(|(_, bytes)| bytes[0])
    }

    fn read_key(
        &self,
        key: &str,
    ) -> Option<(u32, [u8; 32])> {
        let key = fourcc(key)?;
        let info = self.call(&SmcKeyData {
            key,
            data8: SMC_CMD_READ_KEYINFO,
            ..Default::default()
        })?;
        let value = self.call(&SmcKeyData {
            key,
            key_info: SmcKeyInfo {
                data_size: info.key_info.data_size,
                ..Default::default()
            },
            data8: SMC_CMD_READ_BYTES,
            ..Default::default()
        })?;
        Some((info.key_info.data_type, value.bytes))
    }

    fn call(
        &self,
        input: &SmcKeyData,
    ) -> Option<SmcKeyData> {
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
        (result == 0 && output.result == 0).then_some(output)
    }
}

impl Drop for Smc {
    fn drop(&mut self) {
        IOServiceClose(self.connection);
    }
}
