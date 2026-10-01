use std::ptr::null_mut;

const COMPRESSION_LZFSE: i32 = 0x801;
const LZFSE_MAX_OVERHEAD: usize = 12;

#[link(name = "compression")]
unsafe extern "C" {
    fn compression_encode_buffer(
        dst_buffer: *mut u8,
        dst_size: usize,
        src_buffer: *const u8,
        src_size: usize,
        scratch_buffer: *mut u8,
        algorithm: i32,
    ) -> usize;
}

type SizeHeader = u32;

pub fn compress(src: &[u8]) -> Vec<u8> {
    let max_output_body_size = src.len() + LZFSE_MAX_OVERHEAD;
    let mut output = Vec::with_capacity(size_of::<SizeHeader>() + max_output_body_size);
    output.extend((src.len() as SizeHeader).to_le_bytes());

    unsafe {
        let output_body_size = compression_encode_buffer(
            output.as_mut_ptr().byte_add(size_of::<SizeHeader>()),
            max_output_body_size,
            src.as_ptr(),
            src.len(),
            null_mut(),
            COMPRESSION_LZFSE,
        );
        assert!(output_body_size != 0, "compression failed");
        output.set_len(size_of::<SizeHeader>() + output_body_size);
    };

    output
}
