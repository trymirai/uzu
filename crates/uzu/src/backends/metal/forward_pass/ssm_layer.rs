use bytemuck::fill_zeroes;

use super::kv_cache_layer::ArrayCell;
use crate::device::array::Array;

#[derive(Debug)]
pub struct SSMLayer {
    pub conv_state: ArrayCell,
    pub ssm_state: ArrayCell,
    // Staging for SSM per-suffix intermediates
    pub packed: ArrayCell, // [batch_size, conv_dim]
    pub x: ArrayCell,      // [batch_size, num_heads, head_dim]
    pub b: ArrayCell,      // [batch_size, num_groups, state_dim]
    pub c: ArrayCell,      // [batch_size, num_groups, state_dim]
    pub dt: ArrayCell,     // [batch_size, num_heads]
    pub decay: ArrayCell,  // [batch_size, num_heads]
    pub z: ArrayCell,      // [batch_size, num_heads, head_dim]
}

impl SSMLayer {
    pub fn zero(&self) {
        {
            let mut conv = self.conv_state.borrow_mut();
            fill_zeroes(conv.buffer_mut());
        }
        {
            let mut ssm = self.ssm_state.borrow_mut();
            fill_zeroes(ssm.buffer_mut());
        }
        {
            let mut p = self.packed.borrow_mut();
            fill_zeroes(p.buffer_mut());
        }
        {
            let mut x = self.x.borrow_mut();
            fill_zeroes(x.buffer_mut());
        }
        {
            let mut b = self.b.borrow_mut();
            fill_zeroes(b.buffer_mut());
        }
        {
            let mut c = self.c.borrow_mut();
            fill_zeroes(c.buffer_mut());
        }
        {
            let mut dt = self.dt.borrow_mut();
            fill_zeroes(dt.buffer_mut());
        }
        {
            let mut decay = self.decay.borrow_mut();
            fill_zeroes(decay.buffer_mut());
        }
        {
            let mut z = self.z.borrow_mut();
            fill_zeroes(z.buffer_mut());
        }
    }
}
