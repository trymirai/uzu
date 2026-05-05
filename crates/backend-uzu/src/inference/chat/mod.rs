mod bridging;
mod state;

use std::{path::PathBuf, pin::Pin};

use bridging::{build_decoding_config, build_input_and_run_config, build_output};
use futures::{
    Stream,
    channel::mpsc::{self, UnboundedSender},
};
use shoji::{
    traits::{
        Instance as InstanceTrait, State as StateTrait,
        backend::{
            Error as BackendError,
            chat_message::{Output as ChatMessageOutput, StreamInput, StreamOutput},
        },
    },
    types::session::chat::{ChatConfig, ChatReplyConfig},
};
use state::State;
use tokio_util::sync::CancellationToken;

use crate::{
    inference::{Container, Error},
    session::{
        ChatSession,
        config::RunConfig,
        types::{Input, Output},
    },
};

pub struct Instance {
    session_container: Container<ChatSession>,
}

impl Instance {
    pub fn new(
        reference: String,
        config: ChatConfig,
    ) -> Result<Self, Error> {
        let model_path = PathBuf::from(reference.clone());
        let decoding_config = build_decoding_config(&config, &model_path)?;
        let session = ChatSession::new(model_path, decoding_config).map_err(Error::from)?;
        let container = Container::new(session);
        Ok(Self {
            session_container: container,
        })
    }
}

impl InstanceTrait for Instance {
    type StreamConfig = ChatReplyConfig;
    type StreamInput = StreamInput;
    type StreamOutput = StreamOutput;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn StateTrait>, BackendError>> + Send + '_>> {
        let result: Result<(), Error> = self
            .session_container
            .value
            .lock()
            .map_err(|error| Error::Runtime {
                message: error.to_string(),
            })
            .and_then(|mut session| session.reset().map_err(Error::from));
        Box::pin(async move {
            result.map_err(|error| Box::new(error) as BackendError)?;
            Ok(Box::new(State) as Box<dyn StateTrait>)
        })
    }

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        _state: &'a mut dyn StateTrait,
        config: Self::StreamConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, BackendError>> + Send + 'a>> {
        let session = self.session_container.clone();
        let messages = input.clone();
        let cancel_token_for_run = cancel_token.clone();

        let (sender, receiver) = mpsc::unbounded::<Result<ChatMessageOutput, BackendError>>();
        let (input, run_config) = build_input_and_run_config(&messages, &config);
        std::thread::spawn(move || {
            run(session, input, run_config, cancel_token_for_run, sender);
        });
        Box::pin(receiver)
    }
}

fn run(
    session_container: Container<ChatSession>,
    uzu_input: Input,
    run_config: RunConfig,
    cancel_token: CancellationToken,
    sender: UnboundedSender<Result<ChatMessageOutput, BackendError>>,
) {
    let progress = {
        let sender = sender.clone();
        let cancel_token = cancel_token.clone();
        move |output: Output| -> bool {
            let chat_output = build_output(&output);
            let _ = sender.unbounded_send(Ok(chat_output));
            !cancel_token.is_cancelled()
        }
    };

    let result = {
        let mut session = match session_container.value.lock() {
            Ok(session) => session,
            Err(error) => {
                let _ = sender.unbounded_send(Err(Box::new(Error::Runtime {
                    message: error.to_string(),
                }) as BackendError));
                return;
            },
        };
        session.run(uzu_input, run_config, Some(progress))
    };

    let result = match result {
        Ok(output) => Ok(build_output(&output)),
        Err(error) => Err(Box::new(Error::from(error)) as BackendError),
    };
    let _ = sender.unbounded_send(result);
}
