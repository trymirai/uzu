use metal::MTLDevice;
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::common::{Backend, Device};

use super::MetalBackend;

pub struct MetalDevice {
    pub(super) mtl_device: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl Device for MetalDevice {
    type Backend = MetalBackend;

    fn open() -> Result<Self, <Self::Backend as Backend>::Error> {
        let mtl_device = <dyn metal::MTLDevice>::system_default()
            .ok_or(<Self::Backend as Backend>::Error)?;

        Ok(MetalDevice {
            mtl_device,
        })
    }

    fn create_context(
        &self
    ) -> Result<
        <Self::Backend as Backend>::Context,
        <Self::Backend as Backend>::Error,
    > {
        todo!()
    }
}
