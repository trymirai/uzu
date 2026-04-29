use std::{ops::Deref, ptr::NonNull};

use block2::{Block, RcBlock};
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSURLSessionDataTask, NSURLSessionDownloadTask, NSURLSessionUploadTask};

pub struct AppleGetTasksHandler(
    RcBlock<
        dyn Fn(
            NonNull<NSArray<NSURLSessionDataTask>>,
            NonNull<NSArray<NSURLSessionUploadTask>>,
            NonNull<NSArray<NSURLSessionDownloadTask>>,
        ),
    >,
);

impl AppleGetTasksHandler {
    pub fn new(
        handler: impl Fn(
            Box<[Retained<NSURLSessionDataTask>]>,
            Box<[Retained<NSURLSessionUploadTask>]>,
            Box<[Retained<NSURLSessionDownloadTask>]>,
        ) + 'static
    ) -> Self {
        Self(RcBlock::new(
            move |data_tasks_pointer: NonNull<NSArray<NSURLSessionDataTask>>,
                  upload_tasks_pointer: NonNull<NSArray<NSURLSessionUploadTask>>,
                  download_tasks_pointer: NonNull<NSArray<NSURLSessionDownloadTask>>| {
                let data_tasks = unsafe { data_tasks_pointer.as_ref() };
                let upload_tasks = unsafe { upload_tasks_pointer.as_ref() };
                let download_tasks = unsafe { download_tasks_pointer.as_ref() };

                handler(
                    data_tasks.to_vec().into_boxed_slice(),
                    upload_tasks.to_vec().into_boxed_slice(),
                    download_tasks.to_vec().into_boxed_slice(),
                );
            },
        ))
    }
}

impl Deref for AppleGetTasksHandler {
    type Target = Block<
        dyn Fn(
            NonNull<NSArray<NSURLSessionDataTask>>,
            NonNull<NSArray<NSURLSessionUploadTask>>,
            NonNull<NSArray<NSURLSessionDownloadTask>>,
        ),
    >;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
