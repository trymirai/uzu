import definitions;

[[Variants("T", "uint, uint8_t")]]
[[Kernel]]
func BlitCopy<T>(
  Ptr<T, Access.Read> source,
  Ptr<T, Access.ReadWrite> destination,
  uint length,
  [[Axis("length", "256")]] uint position,
) {
  destination[position] = source[position];
}
