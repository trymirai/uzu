use crate::prelude::*;

/// A wrapper for NSURLSession getTasksWithCompletionHandler block.
///
/// Signature corresponds to Apple's API:
/// (NSArray<NSURLSessionDataTask>*, NSArray<NSURLSessionUploadTask>*, NSArray<NSURLSessionDownloadTask>*) -> void
pub struct URLSessionGetTasksCompletionHandler(
    RcBlock<
        dyn Fn(
            NonNull<NSArray<NSURLSessionDataTask>>,
            NonNull<NSArray<NSURLSessionUploadTask>>,
            NonNull<NSArray<NSURLSessionDownloadTask>>,
        ),
    >,
);

impl URLSessionGetTasksCompletionHandler {
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(
                Box<[Retained<NSURLSessionDataTask>]>,
                Box<[Retained<NSURLSessionUploadTask>]>,
                Box<[Retained<NSURLSessionDownloadTask>]>,
            ) + 'static,
    {
        Self(RcBlock::new(
            move |data_tasks_ptr: NonNull<NSArray<NSURLSessionDataTask>>,
                  upload_tasks_ptr: NonNull<NSArray<NSURLSessionUploadTask>>,
                  download_tasks_ptr: NonNull<NSArray<NSURLSessionDownloadTask>>| {
                let data_tasks = unsafe { data_tasks_ptr.as_ref() };
                let upload_tasks = unsafe { upload_tasks_ptr.as_ref() };
                let download_tasks = unsafe { download_tasks_ptr.as_ref() };

                let data_vec = data_tasks.to_vec();
                let upload_vec = upload_tasks.to_vec();
                let download_vec = download_tasks.to_vec();

                handler(data_vec.into_boxed_slice(), upload_vec.into_boxed_slice(), download_vec.into_boxed_slice());
            },
        ))
    }
}

impl Deref for URLSessionGetTasksCompletionHandler {
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
