#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct SmcVersion {
    major: u8,
    minor: u8,
    build: u8,
    reserved: u8,
    release: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct SmcLimitData {
    version: u16,
    length: u16,
    cpu_plimit: u32,
    gpu_plimit: u32,
    mem_plimit: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct SmcKeyInfo {
    pub(super) data_size: u32,
    pub(super) data_type: u32,
    pub(super) data_attributes: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct SmcKeyData {
    pub(super) key: u32,
    pub(super) version: SmcVersion,
    pub(super) limit: SmcLimitData,
    pub(super) key_info: SmcKeyInfo,
    pub(super) result: u8,
    pub(super) status: u8,
    pub(super) data8: u8,
    pub(super) data32: u32,
    pub(super) bytes: [u8; 32],
}

const _: () = assert!(core::mem::size_of::<SmcKeyData>() == 80);
