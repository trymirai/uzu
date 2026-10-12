use bytemuck::NoUninit;

#[derive(Clone, Copy, Debug, NoUninit)]
#[repr(C)]
pub struct Copy {
    pub source: u32,
    pub destination: u32,
}
