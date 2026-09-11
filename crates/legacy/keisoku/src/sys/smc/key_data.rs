#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SmcVersion {
    major: u8,
    minor: u8,
    build: u8,
    reserved: u8,
    release: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SmcLimitData {
    version: u16,
    length: u16,
    cpu_plimit: u32,
    gpu_plimit: u32,
    mem_plimit: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SmcKeyInfo {
    pub data_size: u32,
    pub data_type: u32,
    pub data_attributes: u8,
}

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
