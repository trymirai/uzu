use std::{ffi::CString, mem::MaybeUninit};

use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;

use crate::{
    config::token_codec::chat_codec::ChatCodecConfig,
    session::{
        parameter::ConfigResolvableValue,
        types::{Error, Input, Message, Role},
    },
};

pub trait InputProcessor: Send + Sync {
    fn process(
        &self,
        input: &Input,
        enable_thinking: bool,
        add_generation_prompt: bool,
    ) -> Result<String, Error>;
}

pub struct InputProcessorDefault {
    token_codec_config: ChatCodecConfig,
}

impl InputProcessorDefault {
    pub fn new(token_codec_config: ChatCodecConfig) -> Self {
        Self {
            token_codec_config,
        }
    }
}

/// Formats the current local time according to the provided `strftime` pattern.
fn strftime_now(format_string: String) -> String {
    let Ok(c_format) = CString::new(format_string) else {
        return String::new();
    };

    unsafe {
        let now = libc::time(std::ptr::null_mut());
        if now == -1 {
            return String::new();
        }

        let mut tm = MaybeUninit::<libc::tm>::uninit();
        if libc::localtime_r(&now, tm.as_mut_ptr()).is_null() {
            return String::new();
        }

        let tm = tm.assume_init();
        let mut buf_len = 128_usize;

        loop {
            let mut buffer = vec![0u8; buf_len];
            let written =
                libc::strftime(buffer.as_mut_ptr() as *mut libc::c_char, buffer.len(), c_format.as_ptr(), &tm);

            if written > 0 {
                buffer.truncate(written as usize);
                return String::from_utf8_lossy(&buffer).into_owned();
            }

            if buf_len >= 4096 {
                return String::new();
            }

            buf_len *= 2;
        }
    }
}

impl InputProcessor for InputProcessorDefault {
    fn process(
        &self,
        input: &Input,
        enable_thinking: bool,
        add_generation_prompt: bool,
    ) -> Result<String, Error> {
        let mut messages = input.get_messages();
        for message in &messages {
            if message.role != Role::Assistant && message.reasoning_content.is_some() {
                return Err(Error::UnexpectedReasoningContent);
            }
        }
        if let Some(default_system_prompt) = self.token_codec_config.default_system_prompt.clone()
            && messages.first().is_none_or(|message| message.role != Role::System)
        {
            messages.insert(0, Message::system(default_system_prompt));
        }

        let messages =
            messages.into_iter().map(|message| message.resolve(&self.token_codec_config)).collect::<Vec<_>>();
        let template = self.token_codec_config.prompt_template.clone();
        let bos_token = self.token_codec_config.bos_token.clone();
        let eos_token = self.token_codec_config.eos_token.clone();

        let template_name = "chat_template";
        let mut environment = Environment::new();
        environment.set_unknown_method_callback(unknown_method_callback);
        environment.add_function("strftime_now", strftime_now);
        environment.add_template(template_name, template.as_str()).map_err(Error::UnableToLoadPromptTemplate)?;
        let template = environment.get_template(template_name).map_err(Error::UnableToLoadPromptTemplate)?;

        let result = template
            .render(context!(
                messages => messages,
                add_generation_prompt => add_generation_prompt,
                bos_token => bos_token,
                eos_token => eos_token,
                enable_thinking => enable_thinking
            ))
            .map_err(Error::UnableToRenderPromptTemplate)?;
        Ok(result)
    }
}
