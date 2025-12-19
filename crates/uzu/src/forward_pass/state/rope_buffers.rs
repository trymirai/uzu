use std::{cell::RefCell, rc::Rc};

use super::super::ModelShape;
use crate::{Array, DeviceContext, parameters::ParameterTree};

type ArrayCell<C> = RefCell<<C as DeviceContext>::DeviceArray>;

pub struct RopeBuffers<C: DeviceContext> {
    /// [rope_max_sequence_length, head_dim]
    pub cosines: ArrayCell<C>,
    /// [rope_max_sequence_length, head_dim]
    pub sines: ArrayCell<C>,
}

impl<C: DeviceContext> RopeBuffers<C> {
    pub fn new(
        context: &C,
        model_shape: &ModelShape,
    ) -> Self {
        unsafe {
            let rotated_queries_shape = model_shape.rotated_queries_shape(1);
            let head_dim = rotated_queries_shape[2];
            let rope_max_sequence_length = model_shape.context_length();

            Self {
                cosines: RefCell::new(context.array_uninitialized(
                    &[rope_max_sequence_length, head_dim],
                    model_shape.activation_data_type(),
                )),
                sines: RefCell::new(context.array_uninitialized(
                    &[rope_max_sequence_length, head_dim],
                    model_shape.activation_data_type(),
                )),
            }
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<Rc<C>>,
        rope_name: String,
    ) {
        let Ok(rope_tree) = parameter_tree.subtree(rope_name.as_str()) else {
            return;
        };

        let cosines_view = rope_tree.leaf("cosines").unwrap();
        self.cosines.borrow_mut().copy_from(&cosines_view);

        let sines_view = rope_tree.leaf("sines").unwrap();
        self.sines.borrow_mut().copy_from(&sines_view);
    }
}
