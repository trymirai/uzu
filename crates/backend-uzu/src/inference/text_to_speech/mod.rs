#[cfg(metal_backend)]
mod bridging;
mod state;

use std::pin::Pin;

#[cfg(metal_backend)]
use bridging::{build_input, build_pcm_batch};
use futures::Stream;
#[cfg(not(metal_backend))]
use futures::stream;
#[cfg(metal_backend)]
use futures::{
    StreamExt,
    channel::mpsc::{self, UnboundedSender},
};
#[cfg(metal_backend)]
use rand::prelude::*;
use shoji::traits::{
    Instance as InstanceTrait, State as StateTrait,
    backend::{
        Error as BackendError,
        text_to_speech::{StreamInput, StreamOutput},
    },
};
#[cfg(metal_backend)]
use shoji::types::basic::PcmBatch as ShojiPcmBatch;
use state::State;
use tokio_util::sync::CancellationToken;

use crate::inference::Error;
#[cfg(metal_backend)]
use crate::{
    backends::metal::Metal,
    inference::Container,
    session::{TtsSession, config::TtsRunConfig, types::Input},
};

#[cfg(metal_backend)]
type UzuTtsSession = TtsSession<Metal>;

pub struct Instance {
    #[cfg(metal_backend)]
    session_container: Container<UzuTtsSession>,
}

impl Instance {
    #[cfg(metal_backend)]
    pub fn new(reference: String) -> Result<Self, Error> {
        let model_path = std::path::PathBuf::from(reference);
        let session = UzuTtsSession::new(model_path).map_err(Error::from)?;
        Ok(Self {
            session_container: Container::new(session),
        })
    }

    #[cfg(not(metal_backend))]
    pub fn new(_reference: String) -> Result<Self, Error> {
        Err(Error::Runtime {
            message: "Text-to-speech requires the metal backend".to_string(),
        })
    }
}

impl InstanceTrait for Instance {
    type StreamConfig = ();
    type StreamInput = StreamInput;
    type StreamOutput = StreamOutput;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn StateTrait>, BackendError>> + Send + '_>> {
        Box::pin(async move { Ok(Box::new(State) as Box<dyn StateTrait>) })
    }

    #[cfg(metal_backend)]
    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        _state: &'a mut dyn StateTrait,
        _config: Self::StreamConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, BackendError>> + Send + 'a>> {
        let input = build_input(input);
        let session_container = self.session_container.clone();
        let cancel_for_run = cancel_token.clone();
        let (sender, receiver) = mpsc::unbounded::<Result<ShojiPcmBatch, BackendError>>();

        std::thread::spawn(move || {
            run(session_container, input, cancel_for_run, sender);
        });

        Box::pin(receiver.take_until(cancel_token.cancelled_owned()))
    }

    #[cfg(not(metal_backend))]
    fn stream<'a>(
        &'a self,
        _input: &'a Self::StreamInput,
        _state: &'a mut dyn StateTrait,
        _config: Self::StreamConfig,
        _cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, BackendError>> + Send + 'a>> {
        Box::pin(stream::once(async {
            Err(Box::new(Error::Runtime {
                message: "Text-to-speech requires the metal backend".to_string(),
            }) as BackendError)
        }))
    }
}

#[cfg(metal_backend)]
fn run(
    session_container: Container<UzuTtsSession>,
    input: Input,
    _cancel_token: CancellationToken,
    sender: UnboundedSender<Result<ShojiPcmBatch, BackendError>>,
) {
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
        let seed = rand::rng().random::<u64>();
        let config = TtsRunConfig::default();
        let chunk_sender = sender.clone();
        session.synthesize_streaming_with_seed_and_config(input, seed, &config, |pcm| {
            let _ = chunk_sender.unbounded_send(Ok(build_pcm_batch(pcm)));
        })
    };
    if let Err(error) = result {
        let _ = sender.unbounded_send(Err(Box::new(Error::from(error)) as BackendError));
    }
}
