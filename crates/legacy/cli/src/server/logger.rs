#[derive(Clone)]
pub struct Logger {
    console: bool,
    file: bool,
}

impl Logger {
    pub fn new(
        console: bool,
        file: bool,
    ) -> Self {
        Self {
            console,
            file,
        }
    }

    pub fn msg(
        &self,
        msg: impl Into<String>,
    ) {
        let msg_str = msg.into();
        if self.console {
            println!("{msg_str}")
        }
    }

    pub fn err(
        &self,
        msg: impl Into<String>,
    ) {
        let msg_str = msg.into();
        if self.console {
            eprintln!("{msg_str}",)
        }
    }
}

impl Default for Logger {
    fn default() -> Self {
        Logger::new(true, true)
    }
}
