#pragma once

#include <metal_stdlib>
using namespace metal;

template <typename T>
struct TensorView2D {
  device T* data;
  uint dim0, dim1;
  uint stride0, stride1;

  TensorView2D(device T* ptr) : data(ptr) {}

  TensorView2D(device T* ptr, int d0, int d1) : data(ptr), dim0(d0), dim1(d1) {
    stride0 = d1;
    stride1 = 1;
  }

  TensorView2D(device T* ptr, int d0, int d1, int s0, int s1)
      : data(ptr), dim0(d0), dim1(d1), stride0(s0), stride1(s1) {}

  thread TensorView2D& shaped(int d0, int d1) {
    dim0 = d0;
    dim1 = d1;
    stride0 = d1;
    stride1 = 1;
    return *this;
  }

  device T& at(uint i, uint j) const { return data[i * stride0 + j * stride1]; }

  device T& operator()(uint i, uint j) const { return at(i, j); }
};

template <typename T>
struct TensorView3D {
  device T* data;
  uint dim0, dim1, dim2;
  uint stride0, stride1, stride2;

  TensorView3D(device T* ptr) : data(ptr) {}

  TensorView3D(device T* ptr, int d0, int d1, int d2)
      : data(ptr), dim0(d0), dim1(d1), dim2(d2) {
    stride0 = d1 * d2;
    stride1 = d2;
    stride2 = 1;
  }

  TensorView3D(device T* ptr, int d0, int d1, int d2, int s0, int s1, int s2)
      : data(ptr), dim0(d0), dim1(d1), dim2(d2), stride0(s0), stride1(s1),
        stride2(s2) {}

  thread TensorView3D& shaped(int d0, int d1, int d2) {
    dim0 = d0;
    dim1 = d1;
    dim2 = d2;
    stride0 = d1 * d2;
    stride1 = d2;
    stride2 = 1;
    return *this;
  }

  device T& at(uint i, uint j, uint k) const {
    return data[i * stride0 + j * stride1 + k * stride2];
  }

  device T& operator()(uint i, uint j, uint k) const { return at(i, j, k); }
};
