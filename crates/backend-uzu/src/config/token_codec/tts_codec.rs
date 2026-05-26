use proc_macros::uzu_config;

#[uzu_config(super::TokenCodecConfig)]
pub struct TTSCodecConfig {
    pub prompt_template: String,
    pub drop_initial_newline: bool,
}
