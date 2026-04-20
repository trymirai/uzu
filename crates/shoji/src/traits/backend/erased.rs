pub type DynError = Box<dyn std::error::Error + Send + Sync>;

#[macro_export]
macro_rules! erased_backend_family {
    ($module:ident) => {
        pub mod $module {
            use std::{any::Any, pin::Pin};

            use futures::{Stream, StreamExt, TryStreamExt};
            use $crate::traits::backend::{
                self,
                erased::DynError,
                $module::{Input, Output},
            };

            pub trait AnyBackend: std::fmt::Debug + Send + Sync + 'static {
                fn identifier(&self) -> String;
                fn version(&self) -> String;
            }

            pub trait AnyBackendInstance: Send + Sync + 'static {
                fn identifier(&self) -> String;
                fn version(&self) -> String;

                fn load_model(
                    &self,
                    reference: String,
                ) -> Pin<Box<dyn std::future::Future<Output = Result<Box<dyn AnyLoadedModel>, DynError>> + Send + '_>>;
            }

            pub trait AnyLoadedModel: Send + Sync + 'static {
                fn new_state(
                    &self
                ) -> Pin<
                    Box<dyn std::future::Future<Output = Result<Box<dyn AnyLoadedModelState>, DynError>> + Send + '_>,
                >;

                fn stream<'a>(
                    &'a self,
                    input: &'a Input,
                    state: &'a mut dyn AnyLoadedModelState,
                ) -> Pin<Box<dyn Stream<Item = Result<Output, DynError>> + 'a>>;
            }

            pub trait AnyLoadedModelState: Send + Sync + 'static {
                fn clone_boxed(&self) -> Box<dyn AnyLoadedModelState>;
                fn as_any(&self) -> &dyn Any;
                fn as_any_mut(&mut self) -> &mut dyn Any;
            }

            impl<T> AnyBackend for T
            where
                T: backend::Backend<StreamInput = Input, StreamOutput = Output> + Send + Sync,
            {
                fn identifier(&self) -> String {
                    backend::Backend::identifier(self)
                }
                fn version(&self) -> String {
                    backend::Backend::version(self)
                }
            }

            impl<T> AnyBackendInstance for T
            where
                T: backend::BackendInstance + Send + Sync + 'static,
                T::Backend: backend::Backend<StreamInput = Input, StreamOutput = Output>,
                <T::Backend as backend::Backend>::LoadedModel: AnyLoadedModel,
                <T::Backend as backend::Backend>::Error: Send + Sync + 'static,
            {
                fn identifier(&self) -> String {
                    backend::Backend::identifier(&backend::BackendInstance::backend(self))
                }

                fn version(&self) -> String {
                    backend::Backend::version(&backend::BackendInstance::backend(self))
                }

                fn load_model(
                    &self,
                    reference: String,
                ) -> Pin<Box<dyn std::future::Future<Output = Result<Box<dyn AnyLoadedModel>, DynError>> + Send + '_>>
                {
                    Box::pin(async move {
                        backend::BackendInstance::load_model(self, reference)
                            .await
                            .map(|m| Box::new(m) as Box<dyn AnyLoadedModel>)
                            .map_err(|e| Box::new(e) as DynError)
                    })
                }
            }

            impl<T> AnyLoadedModel for T
            where
                T: backend::LoadedModel + Send + Sync + 'static,
                T::Backend: backend::Backend<StreamInput = Input, StreamOutput = Output>,
                <T::Backend as backend::Backend>::LoadedModelState: AnyLoadedModelState,
                <T::Backend as backend::Backend>::Error: Send + Sync + 'static,
            {
                fn new_state(
                    &self
                ) -> Pin<
                    Box<dyn std::future::Future<Output = Result<Box<dyn AnyLoadedModelState>, DynError>> + Send + '_>,
                > {
                    Box::pin(async move {
                        backend::LoadedModel::new_state(self)
                            .await
                            .map(|s| Box::new(s) as Box<dyn AnyLoadedModelState>)
                            .map_err(|e| Box::new(e) as DynError)
                    })
                }

                fn stream<'a>(
                    &'a self,
                    input: &'a Input,
                    state: &'a mut dyn AnyLoadedModelState,
                ) -> Pin<Box<dyn Stream<Item = Result<Output, DynError>> + 'a>> {
                    let typed_state =
                        match state.as_any_mut().downcast_mut::<<T::Backend as backend::Backend>::LoadedModelState>() {
                            Some(state) => state,
                            None => {
                                let error: DynError = "State type mismatch".into();
                                return Box::pin(futures::stream::iter([Err(error)]));
                            },
                        };

                    let future_stream = backend::LoadedModel::stream(self, input, typed_state);
                    let boxed_future_stream = Box::pin(async move {
                        let inner_stream: Pin<
                            Box<dyn Stream<Item = Result<Output, <T::Backend as backend::Backend>::Error>>>,
                        > = Box::pin(future_stream.await.into_stream());
                        inner_stream
                    });
                    Box::pin(futures::stream::once(boxed_future_stream).flatten().map(
                        |result: Result<Output, <T::Backend as backend::Backend>::Error>| {
                            result.map_err(|error| Box::new(error) as DynError)
                        },
                    ))
                }
            }

            impl<T> AnyLoadedModelState for T
            where
                T: backend::LoadedModelState + Send + Sync + 'static,
                T::Backend: backend::Backend<StreamInput = Input, StreamOutput = Output>,
            {
                fn clone_boxed(&self) -> Box<dyn AnyLoadedModelState> {
                    Box::new(self.clone())
                }
                fn as_any(&self) -> &dyn Any {
                    self
                }
                fn as_any_mut(&mut self) -> &mut dyn Any {
                    self
                }
            }
        }
    };
}

erased_backend_family!(token);
erased_backend_family!(message);

pub enum AnyBackend {
    Token(Box<dyn token::AnyBackendInstance>),
    Message(Box<dyn message::AnyBackendInstance>),
}

impl AnyBackend {
    pub fn identifier(&self) -> String {
        match self {
            AnyBackend::Token(instance) => instance.identifier(),
            AnyBackend::Message(instance) => instance.identifier(),
        }
    }

    pub fn version(&self) -> String {
        match self {
            AnyBackend::Token(instance) => instance.version(),
            AnyBackend::Message(instance) => instance.version(),
        }
    }

    pub async fn load_model(
        &self,
        reference: String,
    ) -> Result<AnyLoadedModel, DynError> {
        match self {
            AnyBackend::Token(instance) => instance.load_model(reference).await.map(AnyLoadedModel::Token),
            AnyBackend::Message(instance) => instance.load_model(reference).await.map(AnyLoadedModel::Message),
        }
    }
}

pub enum AnyLoadedModel {
    Token(Box<dyn token::AnyLoadedModel>),
    Message(Box<dyn message::AnyLoadedModel>),
}

impl AnyLoadedModel {
    pub async fn new_state(&self) -> Result<AnyLoadedModelState, DynError> {
        match self {
            AnyLoadedModel::Token(model) => model.new_state().await.map(AnyLoadedModelState::Token),
            AnyLoadedModel::Message(model) => model.new_state().await.map(AnyLoadedModelState::Message),
        }
    }
}

pub enum AnyLoadedModelState {
    Token(Box<dyn token::AnyLoadedModelState>),
    Message(Box<dyn message::AnyLoadedModelState>),
}

impl AnyLoadedModelState {
    pub fn clone_boxed(&self) -> Self {
        match self {
            AnyLoadedModelState::Token(state) => AnyLoadedModelState::Token(state.clone_boxed()),
            AnyLoadedModelState::Message(state) => AnyLoadedModelState::Message(state.clone_boxed()),
        }
    }
}
