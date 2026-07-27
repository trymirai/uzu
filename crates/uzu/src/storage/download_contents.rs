use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DownloadContents: u8 {
        const CONFIG = 0b0000_0001;
        const TOKENIZER = 0b0000_0010;
        const WEIGHTS = 0b0000_0100;
        const TRACES = 0b0000_1000;
    }
}

impl Default for DownloadContents {
    fn default() -> Self {
        Self::all()
    }
}

impl DownloadContents {
    pub fn includes_file(
        &self,
        file_name: &str,
    ) -> bool {
        let flag = match file_name {
            "config.json" => Self::CONFIG,
            "tokenizer.json" => Self::TOKENIZER,
            "traces.safetensors" => Self::TRACES,
            name if name.ends_with(".safetensors") => Self::WEIGHTS,
            _ => Self::CONFIG,
        };
        self.contains(flag)
    }
}
