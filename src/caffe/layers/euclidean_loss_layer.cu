// Copyright 2014 BVLC and contributors.

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
Dtype EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = coeff_ * dot / bottom[0]->num() / Dtype(2);
  return loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  caffe_gpu_axpby(
      (*bottom)[0]->count(),              // count
      coeff_ / (*bottom)[0]->num(),     // alpha
      diff_.gpu_data(),                   // a
      Dtype(0),                           // beta
      (*bottom)[0]->mutable_gpu_diff());  // b
}

INSTANTIATE_CLASS(EuclideanLossLayer);

}  // namespace caffe
