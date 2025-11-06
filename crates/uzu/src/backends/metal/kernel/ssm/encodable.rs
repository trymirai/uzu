use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    Conv1dUpdateKernel, SSDUpdateArguments, SSDUpdateKernel,
    SSDUpdateNoZArguments, SSDUpdateNoZKernel, SSMKernelError,
};
use crate::backends::metal::{
    KernelDataType, MTLContext,
    forward_pass::{
        ArrayId, ForwardPassState,
        encodable_with_state::{EncodableWithState, EncodingParameters},
    },
};

pub struct SSMLayerKernelEncodable {
    layer_index: usize,
    kernel_size: i32,
    num_heads: usize,
    head_dim: usize,
    num_groups: usize,
    state_dim: usize,
    data_type: KernelDataType,

    // IO mapping
    x_id: ArrayId,         // [b, h*dh] interpreted as (b,h,dh)
    b_id: ArrayId,         // [b, g, n]
    c_id: ArrayId,         // [b, g, n]
    dt_id: ArrayId,        // [b, h]
    decay_id: ArrayId,     // [b, h]
    d_id: ArrayId,         // [h]
    z_id: Option<ArrayId>, // optional [b, h*dh]
    y_id: ArrayId,         // [b, h*dh]

    // Kernels
    ssd_update: SSDUpdateKernel,
    ssd_update_no_z: SSDUpdateNoZKernel,
    _conv_update: Option<Conv1dUpdateKernel>, // reserved for future conv usage
}

impl SSMLayerKernelEncodable {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
        layer_index: usize,
        kernel_size: i32,
        num_heads: usize,
        head_dim: usize,
        num_groups: usize,
        state_dim: usize,
        x_id: ArrayId,
        b_id: ArrayId,
        c_id: ArrayId,
        dt_id: ArrayId,
        decay_id: ArrayId,
        d_id: ArrayId,
        z_id: Option<ArrayId>,
        y_id: ArrayId,
    ) -> Result<Self, SSMKernelError> {
        let ssd_update = SSDUpdateKernel::new(context, data_type)?;
        let ssd_update_no_z = SSDUpdateNoZKernel::new(context, data_type)?;
        Ok(Self {
            layer_index,
            kernel_size,
            num_heads,
            head_dim,
            num_groups,
            state_dim,
            data_type,
            x_id,
            b_id,
            c_id,
            dt_id,
            decay_id,
            d_id,
            z_id,
            y_id,
            ssd_update,
            ssd_update_no_z,
            _conv_update: None,
        })
    }
}

impl EncodableWithState for SSMLayerKernelEncodable {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        _parameters: &EncodingParameters,
    ) {
        let suffix_length = state.aux_buffers_suffix_length();

        // Bind arrays
        let arrays = state.arrays(&[
            self.x_id,
            self.b_id,
            self.c_id,
            self.dt_id,
            self.decay_id,
            self.d_id,
            self.y_id,
            ArrayId::SsmState(self.layer_index),
        ]);

        let z_binding = self.z_id.map(|id| state.arrays(&[id]));

        let mut x_arr = arrays[0].borrow_mut();
        let mut b_arr = arrays[1].borrow_mut();
        let mut c_arr = arrays[2].borrow_mut();
        let mut dt_arr = arrays[3].borrow_mut();
        let mut decay_arr = arrays[4].borrow_mut();
        let mut d_arr = arrays[5].borrow_mut();
        let mut y_arr = arrays[6].borrow_mut();
        let mut ssm_state_arr = arrays[7].borrow_mut();

        let z_buf_opt = z_binding.as_ref().map(|binding| unsafe {
            binding[0].borrow_mut().mtl_buffer().clone()
        });

        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        let compute = mtl_command_buffer.new_compute_command_encoder();

        let x_buf = unsafe { x_arr.mtl_buffer() };
        let b_buf = unsafe { b_arr.mtl_buffer() };
        let c_buf = unsafe { c_arr.mtl_buffer() };
        let dt_buf = unsafe { dt_arr.mtl_buffer() };
        let decay_buf = unsafe { decay_arr.mtl_buffer() };
        let d_buf = unsafe { d_arr.mtl_buffer() };
        let y_buf = unsafe { y_arr.mtl_buffer() };
        let state_buf = unsafe { ssm_state_arr.mtl_buffer() };

        // Compute strides and constants
        let h = self.num_heads;
        let g = self.num_groups;
        let dh = self.head_dim;
        let n = self.state_dim;

        let x_strides = [h * dh, dh, 1usize];
        let dt_strides = [h, 1usize];
        let cb_strides = [g * n, n, 1usize];
        let state_strides = [h * dh * n, dh * n, n, 1usize];
        let group_size = (h / g) as i32;
        let state_size = n as i32;

        // Encode SSD update (per-suffix row is a separate batch element)
        if let Some(z_buf) = z_buf_opt.as_ref() {
            let _ = self.ssd_update.encode(
                &compute,
                SSDUpdateArguments {
                    x: &x_buf,
                    dt: &dt_buf,
                    decay: &decay_buf,
                    b: &b_buf,
                    c: &c_buf,
                    d: &d_buf,
                    z: z_buf,
                    state: &state_buf,
                    y: &y_buf,
                    next_state: &state_buf,
                    group_size,
                    state_size,
                    x_strides,
                    dt_strides,
                    cb_strides,
                    state_strides,
                    b_size: suffix_length,
                    h_size: h,
                    dh_size: dh,
                },
            );
        } else {
            let _ = self.ssd_update_no_z.encode(
                &compute,
                SSDUpdateNoZArguments {
                    x: &x_buf,
                    dt: &dt_buf,
                    decay: &decay_buf,
                    b: &b_buf,
                    c: &c_buf,
                    d: &d_buf,
                    state: &state_buf,
                    y: &y_buf,
                    next_state: &state_buf,
                    group_size,
                    state_size,
                    x_strides,
                    dt_strides,
                    cb_strides,
                    state_strides,
                    b_size: suffix_length,
                    h_size: h,
                    dh_size: dh,
                },
            );
        }

        compute.end_encoding();
    }
}
