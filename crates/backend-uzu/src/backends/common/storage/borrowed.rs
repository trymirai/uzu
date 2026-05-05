use std::marker::PhantomData;

use super::Storage;

pub struct Borrowed<'a, B>(PhantomData<&'a B>);

impl<'a, B> Storage for Borrowed<'a, B> {
    type Buffer = &'a B;
}
