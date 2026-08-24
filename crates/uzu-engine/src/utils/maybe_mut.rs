use std::ops::Deref;

pub enum MaybeMut<'a, T: ?Sized> {
    Const(&'a T),
    Mut(&'a mut T),
}

impl<'a, T: ?Sized> Deref for MaybeMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        match self {
            Self::Const(value) => value,
            Self::Mut(value) => value,
        }
    }
}
