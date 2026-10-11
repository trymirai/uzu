import definitions;
import common.activation;
import common.storage_address;

// exact_quotient on raw FP32 bits: each of the `count` rows of pairs holds a and b, and results gets a / b rounded to
// nearest even.
[[Kernel]] [[Test]] func TestExactQuotient(
  Ptr<uint, Access.Read> pairs,
  Ptr<uint, Access.ReadWrite> results,
  uint count,
  [[Axis("count", "64")]] uint index,
) {
  if (index < count) {
    let row = element_pointer(pairs, element_index(index, 2, 0));
    let result = element_pointer(results, element_index(index, 1, 0));
    result[0] = asuint(exact_quotient(asfloat(row[0]), asfloat(row[1])));
  }
}
