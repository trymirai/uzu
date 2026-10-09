import definitions;
import common.activation;

// Physical storage addresses as two 32-bit words (x = low, y = high), rebuilt into typed pointers by OpBitcast.
// Callers pass a base aligned to T and elements inside one complete, valid buffer span; nothing here checks bounds.
// The modulo-2^64 arithmetic is exact for synthetic words only and never licenses wrapping a live address.

uint2 address_words<T, Access access>(Ptr<T, access> pointer) {
  return spirv_asm {
    result:$$uint2 = OpBitcast $pointer;
  };
}

Ptr<T, access> address_pointer<T, Access access>(uint2 words) {
  return spirv_asm {
    result:$$Ptr<T, access> = OpBitcast $words;
  };
}

// The natural stride that scales element indexes is a compiler intrinsic: a changed storage width fails the build at
// every element_pointer instantiation instead of moving addresses.
void storage_widths() {
  static_assert(__naturalStrideOf<float>() == 4, "storage_address: float is not 4 bytes");
  static_assert(__naturalStrideOf<half>() == 2, "storage_address: half is not 2 bytes");
  static_assert(__naturalStrideOf<bf16>() == 2, "storage_address: bf16 is not 2 bytes");
  static_assert(__naturalStrideOf<uint>() == 4, "storage_address: uint is not 4 bytes");
}

// row * stride + column, exact for every u32 input: at most 2^64 - 2^32.
public uint2 element_index(uint row, uint stride, uint column) {
  let product = multiply_extended(row, stride);
  let low = product.x + column;
  return uint2(low, product.y + (low < column ? 1u : 0u));
}

// element * element_bytes modulo 2^64.
public uint2 byte_offset(uint2 element, uint element_bytes) {
  let low = multiply_extended(element.x, element_bytes);
  return uint2(low.x, element.y * element_bytes + low.y);
}

// base + bytes modulo 2^64.
public uint2 add_address(uint2 base, uint2 bytes) {
  let low = base.x + bytes.x;
  return uint2(low, base.y + bytes.y + (low < bytes.x ? 1u : 0u));
}

// The element `element` past `base`.
public Ptr<T, access> element_pointer<T, Access access>(Ptr<T, access> base, uint2 element) {
  storage_widths();
  let bytes = byte_offset(element, uint(__naturalStrideOf<T>()));
  return address_pointer<T, access>(add_address(address_words(base), bytes));
}
