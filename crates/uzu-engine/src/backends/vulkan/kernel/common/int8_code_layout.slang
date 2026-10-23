// The CPU Int8CodeLayout: where activation code `index` of a row is stored. Grouped by weight nibble permutes each 8
// codes [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 4, 1, 5, 2, 6, 3, 7]; sequential keeps them.
public uint int8_code_index(uint index, bool grouped_by_weight_nibble) {
  let offset = index % 8;
  return grouped_by_weight_nibble ? index - offset + offset % 2 * 4 + offset / 2 : index;
}
