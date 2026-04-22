mod bridging;
mod state;

use std::{path::PathBuf, pin::Pin};

use bridging::{build_input, build_output};
use futures::{
    Stream, StreamExt,
    channel::mpsc::{self, UnboundedSender},
};
use shoji::{
    traits::{
        Instance as InstanceTrait, State as StateTrait,
        backend::{
            Error as BackendError,
            classification::{StreamInput, StreamOutput},
        },
    },
    types::session::classification::ClassificationOutput as ShojiOutput,
};
use state::State;
use tokio_util::sync::CancellationToken;

use crate::{
    inference::{Container, Error},
    session::{ClassificationSession, types::Input},
};

pub struct Instance {
    session_container: Container<ClassificationSession>,
}

impl Instance {
    pub fn new(reference: String) -> Result<Self, Error> {
        let model_path = PathBuf::from(reference);
        let session = ClassificationSession::new(model_path).map_err(Error::from)?;
        let container = Container::new(session);
        Ok(Self {
            session_container: container,
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

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        _state: &'a mut dyn StateTrait,
        _config: Self::StreamConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, BackendError>> + Send + 'a>> {
        let input = build_input(input);
        let session_container = self.session_container.clone();
        let (sender, receiver) = mpsc::unbounded::<Result<ShojiOutput, BackendError>>();

        std::thread::spawn(move || {
            run(session_container, input, sender);
        });

        Box::pin(receiver.take_until(cancel_token.cancelled_owned()))
    }
}

fn run(
    session_container: Container<ClassificationSession>,
    input: Input,
    sender: UnboundedSender<Result<ShojiOutput, BackendError>>,
) {
    let result = {
        let mut session = match session_container.value.lock() {
            Ok(guard) => guard,
            Err(error) => {
                let _ = sender.unbounded_send(Err(Box::new(Error::Runtime {
                    message: error.to_string(),
                }) as BackendError));
                return;
            },
        };
        session.classify(input)
    };
    let result = match result {
        Ok(output) => Ok(build_output(&output)),
        Err(error) => Err(Box::new(Error::from(error)) as BackendError),
    };
    let _ = sender.unbounded_send(result);
}
