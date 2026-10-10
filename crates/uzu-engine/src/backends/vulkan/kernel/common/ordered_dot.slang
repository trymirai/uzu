import definitions;
import common.activation;
import common.storage_address;

// The ordered dot from +0 over d ascending of x[d] y[d].
public float ordered_dot<X: IModelFloat, Y: IModelFloat>(Ptr<X, Access.Read> x, Ptr<Y, Access.Read> y, uint dim) {
  var sum = 0.0;
  for (uint d = 0; d < dim; d++) {
    let y_d = element_pointer(y, uint2(d, 0))[0].toFloat();
    sum = ordered_sum(sum, exact_product(element_pointer(x, uint2(d, 0))[0].toFloat(), y_d));
  }
  return sum;
}
