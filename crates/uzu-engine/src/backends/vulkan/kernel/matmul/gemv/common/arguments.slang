import definitions;

public struct GemvOperands<A : IStorageFloat, B : IStorageFloat, D : IStorageFloat> {
  public Ptr<uint, Access.Read> b;
  public Ptr<A, Access.Read> a;
  public Ptr<D, Access.ReadWrite> d;
  public Ptr<B, Access.Read> output_bias;
  public Ptr<uint, Access.Read> gather_indices;
}
