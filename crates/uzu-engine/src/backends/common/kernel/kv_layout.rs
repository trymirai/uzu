use half::bf16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KVLayout {
    pub head_dim: u32,
    pub num_kv_heads: u32,
    pub key_alignment: u32,
    pub backend_opaque_layout: u32,
}

impl KVLayout {
    pub fn new(
        head_dim: u32,
        num_kv_heads: u32,
        key_alignment: u32,
        backend_opaque_layout: u32,
    ) -> Self {
        Self {
            head_dim,
            num_kv_heads,
            key_alignment,
            backend_opaque_layout,
        }
    }

    pub fn code_bytes(
        &self,
        keys: u32,
    ) -> usize {
        keys.next_multiple_of(self.key_alignment) as usize * self.num_kv_heads as usize * self.head_dim as usize
    }

    pub fn scale_bytes(
        &self,
        keys: u32,
    ) -> usize {
        keys.next_multiple_of(self.key_alignment) as usize * self.num_kv_heads as usize * size_of::<bf16>()
    }
}
