use crate::prelude::*;

pub struct URLSessionResumeDataHandler(RcBlock<dyn Fn(*mut NSData)>);

impl URLSessionResumeDataHandler {
    #[allow(unused)]
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(*mut NSData) + 'static,
    {
        Self(RcBlock::new(handler))
    }

    pub fn new_bytes<F>(handler: F) -> Self
    where
        F: Fn(Box<[u8]>) + 'static,
    {
        Self(RcBlock::new(move |ptr: *mut NSData| {
            let boxed: Box<[u8]> = if let Some(nsdata) = unsafe { ptr.as_ref() } {
                nsdata.to_vec().into_boxed_slice()
            } else {
                // Some systems may deliver nil resume data; treat as empty
                Box::new([])
            };
            handler(boxed);
        }))
    }
}

impl Deref for URLSessionResumeDataHandler {
    type Target = Block<dyn Fn(*mut NSData)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
