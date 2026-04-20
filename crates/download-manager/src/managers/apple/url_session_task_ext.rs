use crate::{TaskID, prelude::*};

#[allow(unused)]
pub trait UrlSessionTaskExt {
    fn task_identifier(&self) -> TaskID;
    fn task_description(&self) -> Option<String>;
    fn set_task_description(
        &self,
        s: &str,
    );
    fn url_string(&self) -> Option<String>;
}

impl UrlSessionTaskExt for NSURLSessionTask {
    fn task_identifier(&self) -> TaskID {
        unsafe { msg_send![self, taskIdentifier] }
    }

    fn task_description(&self) -> Option<String> {
        let desc_opt = self.taskDescription();
        desc_opt.map(|d| d.to_string())
    }

    fn set_task_description(
        &self,
        s: &str,
    ) {
        let ns = NSString::from_str(s);
        self.setTaskDescription(Some(&ns));
    }

    fn url_string(&self) -> Option<String> {
        self.originalRequest()
            .or_else(|| self.currentRequest())
            .and_then(|req| req.URL())
            .and_then(|url| url.absoluteString())
            .map(|s| s.to_string())
    }
}
