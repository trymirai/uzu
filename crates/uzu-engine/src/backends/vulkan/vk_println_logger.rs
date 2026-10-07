use super::VkLogger;

pub struct VkPrintlnLogger;
impl VkLogger for VkPrintlnLogger {
    fn v(
        &self,
        msg: &str,
    ) {
        println!("[Verbose]: {msg}")
    }

    fn i(
        &self,
        msg: &str,
    ) {
        println!("[Info]: {msg}")
    }

    fn d(
        &self,
        msg: &str,
    ) {
        println!("[Debug]: {msg}")
    }

    fn w(
        &self,
        msg: &str,
    ) {
        println!("[Warning]: {msg}")
    }

    fn e(
        &self,
        msg: &str,
    ) {
        eprintln!("[Error]: {msg}")
    }
}
