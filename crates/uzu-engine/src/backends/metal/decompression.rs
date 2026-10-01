use std::ptr::null_mut;

const COMPRESSION_LZFSE: i32 = 0x801;

#[link(name = "compression")]
unsafe extern "C" {
    fn compression_decode_buffer(
        dst_buffer: *mut u8,
        dst_size: usize,
        src_buffer: *const u8,
        src_size: usize,
        scratch_buffer: *mut u8,
        algorithm: i32,
    ) -> usize;
}

type SizeHeader = u32;

pub fn decompress(src: &[u8]) -> Vec<u8> {
    let (src_header, src_body) = src.split_at(size_of::<SizeHeader>());
    let size = SizeHeader::from_le_bytes(src_header.try_into().unwrap()) as usize;

    let mut output = Vec::with_capacity(size + 1);
    unsafe {
        let decompressed_size = compression_decode_buffer(
            output.as_mut_ptr(),
            output.capacity(),
            src_body.as_ptr(),
            src_body.len(),
            null_mut(),
            COMPRESSION_LZFSE,
        );
        assert!(decompressed_size > 0 && decompressed_size == size, "decompression failed");
        output.set_len(decompressed_size);
    }

    output
}
