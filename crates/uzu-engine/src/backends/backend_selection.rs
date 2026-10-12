use crate::backends::common::Backend;

pub trait BackendSelection {
    type Output;
    type Error;

    fn select<B: Backend>(self) -> Result<Self::Output, Self::Error>;
}

pub fn select_backend<S: BackendSelection>(
    selection: S,
    _unavailable: S::Error,
) -> Result<S::Output, S::Error> {
    #[cfg(all(target_vendor = "apple", backend = "metal"))]
    {
        selection.select::<crate::backends::metal::Metal>()
    }
    #[cfg(all(not(target_vendor = "apple"), backend = "vulkan"))]
    {
        selection.select::<crate::backends::vulkan::Vulkan>()
    }
    #[cfg(not(any(
        all(target_vendor = "apple", backend = "metal"),
        all(not(target_vendor = "apple"), backend = "vulkan")
    )))]
    {
        let _ = selection;
        Err(_unavailable)
    }
}
