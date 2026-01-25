use super::Backend;

pub trait Device: Sized {
    type Backend: Backend;

    fn open() -> Result<Self, <Self::Backend as Backend>::Error>;

    fn create_context(
        &self
    ) -> Result<
        <Self::Backend as Backend>::Context,
        <Self::Backend as Backend>::Error,
    >;
}
