use std::ffi::CString;
use std::mem::MaybeUninit;

use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;

use crate::{
    config::MessageProcessorConfig,
    session::{
        parameter::ConfigResolvableValue,
        types::{Error, Input, Role},
    },
};

pub trait InputProcessor: Send + Sync {
    fn process(
        &self,
        input: &Input,
        enable_thinking: bool,
    ) -> Result<String, Error>;
}

pub struct InputProcessorDefault {
    message_processing_config: MessageProcessorConfig,
}

impl InputProcessorDefault {
    pub fn new(message_processing_config: MessageProcessorConfig) -> Self {
        Self {
            message_processing_config,
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
            let written = libc::strftime(
                buffer.as_mut_ptr() as *mut libc::c_char,
                buffer.len(),
                c_format.as_ptr(),
                &tm,
            );

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
    ) -> Result<String, Error> {
        let messages = input.get_messages();
        for message in &messages {
            if message.role != Role::Assistant
                && message.reasoning_content.is_some()
            {
                return Err(Error::UnexpectedReasoningContent);
            }
        }

        let messages = messages
            .into_iter()
            .map(|message| message.resolve(&self.message_processing_config))
            .collect::<Vec<_>>();
        let template = self.message_processing_config.prompt_template.clone();
        let bos_token = self.message_processing_config.bos_token.clone();

        let template_name = "chat_template";
        let mut environment = Environment::new();
        environment.set_unknown_method_callback(unknown_method_callback);
        environment.add_function("strftime_now", strftime_now);
        environment
            .add_template(template_name, template.as_str())
            .map_err(|_| Error::UnableToLoadPromptTemplate)?;
        let template = environment
            .get_template(template_name)
            .map_err(|_| Error::UnableToLoadPromptTemplate)?;

        let result = template.render(context!(
            messages => messages,
            add_generation_prompt => true,
            bos_token => bos_token,
            enable_thinking => enable_thinking
        ));

        // For simple text inputs on base models, fall back to BOS + text if template fails
        match result {
            Ok(rendered) => Ok(rendered),
            Err(e) => {
                eprintln!(
                    "[Template Error] Failed to render prompt template: {:?}",
                    e
                );
                eprintln!(
                    "[Template Error] Falling back to simple format for single-message input"
                );
                // If it's a single user message (typical for Input::Text), use simple format
                if messages.len() == 1 {
                    if let Some(content_str) = messages[0].get("content") {
                        let prefix = bos_token
                            .as_ref()
                            .map(|s| s.as_str())
                            .unwrap_or("");
                        Ok(format!("{}{}", prefix, content_str))
                    } else {
                        Err(Error::UnableToRenderPromptTemplate)
                    }
                } else {
                    Err(Error::UnableToRenderPromptTemplate)
                }
            },
        }
    }
}
