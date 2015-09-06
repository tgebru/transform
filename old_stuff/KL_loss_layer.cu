// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {
  
template <typename Dtype>
__global__ void log_ratio(const int n, const Dtype* in1, const Dtype* in2, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = log(max(in1[index], Dtype(FLT_MIN)))-log(max(in2[index], Dtype(FLT_MIN)));
  }
}

template <typename Dtype>
Dtype SoftmaxKLLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_top_vec_[0] = &prob1_;
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  softmax_bottom_vec_[0] = bottom[1];
  softmax_top_vec_[0] = &prob2_;
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  
  const Dtype* prob1_data = prob1_.gpu_data();
  const Dtype* prob2_data = prob2_.gpu_data();
  Dtype* log_data = log_vector_.mutable_gpu_data();
  int num = prob1_.num();
  int count = prob1_.count();
  
  log_ratio<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, prob1_data, prob2_data, log_data);
  
  caffe_gpu_dot<Dtype>(count, log_data, prob1_data, &loss_);
  
  loss_ *= coeff_;
  
  return loss_ / num;
}

template <typename Dtype>
void SoftmaxKLLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff1 = (*bottom)[0]->mutable_gpu_diff();
  Dtype* bottom_diff2 = (*bottom)[1]->mutable_gpu_diff();
  const Dtype* prob1_data = prob1_.gpu_data();
  const Dtype* prob2_data = prob2_.gpu_data();
  const Dtype* log_data = log_vector_.gpu_data();
  int num = prob1_.num();
  int count = prob1_.count();  
  
  caffe_gpu_copy<Dtype>(count, prob2_data, bottom_diff2);
  caffe_gpu_axpy<Dtype>(count, Dtype(-1), prob1_data, bottom_diff2);  
  caffe_gpu_scal(count, coeff_ / num, bottom_diff2);
  
  caffe_gpu_mul<Dtype>(count, log_data, prob1_data, bottom_diff1);
  caffe_gpu_axpy<Dtype>(count, -loss_, prob1_data, bottom_diff1);  
  caffe_gpu_scal(count, coeff_ / num, bottom_diff1);
}

INSTANTIATE_CLASS(SoftmaxKLLossLayer);


}  // namespace caffe
