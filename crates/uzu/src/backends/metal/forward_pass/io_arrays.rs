use std::{cell::RefCell, collections::HashMap};

use mpsgraph::TensorData;
use objc2::rc::Retained;

use super::state::{ArrayId, ForwardPassState};
use crate::backends::metal::array::MetalArray;

#[derive(Clone)]
pub struct IOArrays {
    input_arrays: Box<[ArrayId]>,
    output_arrays: Box<[ArrayId]>,
}

pub struct MPSGraphFeeds {
    pub inputs: Box<[Retained<TensorData>]>,
    pub outputs: Box<[Retained<TensorData>]>,
}

pub struct KernelFeeds {
    pub inputs: Box<[RefCell<MetalArray>]>,
    pub outputs: Box<[RefCell<MetalArray>]>,
}

impl IOArrays {
    pub fn new(
        input_arrays: Box<[ArrayId]>,
        output_arrays: Box<[ArrayId]>,
    ) -> Self {
        Self {
            input_arrays,
            output_arrays,
        }
    }

    fn all_ids(&self) -> Box<[ArrayId]> {
        let mut potentially_duplicated_results: Vec<ArrayId> = self
            .input_arrays
            .iter()
            .copied()
            .chain(self.output_arrays.iter().copied())
            .collect();
        potentially_duplicated_results.sort();
        potentially_duplicated_results.dedup();
        potentially_duplicated_results.into()
    }

    pub unsafe fn get_mpsgraph_feeds(
        &self,
        state: &mut ForwardPassState,
    ) -> MPSGraphFeeds {
        unsafe {
            let all_ids = self.all_ids();
            let all_tensor_datas =
                state.arrays(&all_ids).into_iter().map(|a_rc_refcell| {
                    a_rc_refcell.borrow_mut().to_mps_tensor_data()
                });
            let id_to_result =
                HashMap::<ArrayId, Retained<TensorData>>::from_iter(
                    all_ids.into_iter().zip(all_tensor_datas),
                );
            let inputs: Box<[_]> = self
                .input_arrays
                .iter()
                .map(|id| id_to_result[id].clone())
                .collect();
            let outputs: Box<[_]> = self
                .output_arrays
                .iter()
                .map(|id| id_to_result[id].clone())
                .collect();
            MPSGraphFeeds {
                inputs,
                outputs,
            }
        }
    }

    pub unsafe fn get_kernel_feeds(
        &self,
        state: &mut ForwardPassState,
    ) -> KernelFeeds {
        let inputs: Box<[RefCell<MetalArray>]> =
            state.arrays(&self.input_arrays);
        let outputs: Box<[RefCell<MetalArray>]> =
            state.arrays(&self.output_arrays);

        KernelFeeds {
            inputs,
            outputs,
        }
    }
}
