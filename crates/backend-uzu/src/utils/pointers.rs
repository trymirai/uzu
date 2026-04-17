pub struct SendPtr<T>(pub *const T);
impl<T> SendPtr<T> {
    pub fn as_ptr(self) -> *const T {
        self.0
    }
}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}

pub struct SendPtrMut<T>(pub *mut T);
impl<T> SendPtrMut<T> {
    pub fn as_ptr(self) -> *mut T {
        self.0
    }
}
unsafe impl<T> Send for SendPtrMut<T> {}
unsafe impl<T> Sync for SendPtrMut<T> {}
impl<T> Clone for SendPtrMut<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtrMut<T> {}
