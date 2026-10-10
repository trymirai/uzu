import definitions;
import common.activation;
import common.storage_address;

// exact_fma on raw FP32 bits: each of the `count` rows of triples holds a, b and c, and results gets a * b + c rounded
// once.
[[Kernel]] [[Test]] func TestExactFma(
  Ptr<uint, Access.Read> triples,
  Ptr<uint, Access.ReadWrite> results,
  uint count,
  [[Axis("count", "64")]] uint index,
) {
  if (index < count) {
    let row = element_pointer(triples, element_index(index, 3, 0));
    let result = element_pointer(results, element_index(index, 1, 0));
    result[0] = asuint(exact_fma(asfloat(row[0]), asfloat(row[1]), asfloat(row[2])));
  }
}
