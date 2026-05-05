use std::{cell::RefCell, ops::DerefMut, rc::Rc};

use crate::{
    DataType,
    backends::common::{Backend, Encoder, Kernels, kernel::TensorFinalizeKernel},
    encodable_block::model_extension::ModelExtensionError,
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub struct TensorFinalize<B: Backend> {
    kernel: <B::Kernels as Kernels>::TensorFinalizeKernel,
    scalar: Option<Rc<RefCell<B::Buffer>>>,
}

impl<B: Backend> TensorFinalize<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        scalar: Option<Rc<RefCell<B::Buffer>>>,
    ) -> Result<Self, ModelExtensionError<B>> {
        let kernel = <B::Kernels as Kernels>::TensorFinalizeKernel::new(context, data_type, scalar.is_some())
            .map_err(ModelExtensionError::BackendError)?;
        Ok(Self {
            kernel,
            scalar,
        })
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) {
        let shortcut = state.array(ArrayId::Shortcut);
        let main = state.array(ArrayId::Main);
        let scalar = self.scalar.as_ref().map(|scalar| scalar.borrow());
        self.kernel.encode(
            shortcut.buffer().borrow_mut().deref_mut(),
            main.buffer().borrow_mut().deref_mut(),
            scalar.as_deref(),
            main.num_elements() as u32,
            encoder,
        );
    }
}
