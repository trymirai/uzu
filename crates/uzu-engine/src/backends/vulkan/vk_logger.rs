pub trait VkLogger: Send + Sync {
    fn v(
        &self,
        msg: &str,
    );

    fn i(
        &self,
        msg: &str,
    );

    fn d(
        &self,
        msg: &str,
    );

    fn w(
        &self,
        msg: &str,
    );

    fn e(
        &self,
        msg: &str,
    );
}
