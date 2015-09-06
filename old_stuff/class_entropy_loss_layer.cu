// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
  
template <typename Dtype>
__global__ void gpu_log(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = log(max(in[index], Dtype(FLT_MIN)));
  }
}

template <typename Dtype>
Dtype ClassEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  const Dtype* data = bottom[0]->gpu_data();
  const Dtype* avg_data = avg_.gpu_data();
  Dtype* class_prob_data = class_prob_.mutable_gpu_data();
  Dtype* log_class_prob_data = log_class_prob_.mutable_gpu_data(); 
  
  alpha_ = alpha_ * discount_coeff_ + Dtype(1);
  caffe_gpu_gemv(CblasTrans, num_, channels_, Dtype(1) / alpha_, data, avg_data, (alpha_ - Dtype(1)) / alpha_, class_prob_data);
  
  gpu_log<Dtype><<<CAFFE_GET_BLOCKS(channels_), CAFFE_CUDA_NUM_THREADS>>>(
      channels_, class_prob_data, log_class_prob_data);
  
  Dtype loss;
  caffe_gpu_dot<Dtype>(channels_, log_class_prob_data, class_prob_data, &loss);
  loss *= coeff_;    
  return loss;
}

template <typename Dtype>
void ClassEntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const Dtype* log_class_prob_data = log_class_prob_.gpu_data();
  const Dtype* ones_data = ones_.gpu_data();
  
  for (int i=0; i<num_; i++) {
    caffe_gpu_copy<Dtype>(channels_, log_class_prob_data, bottom_diff);
    caffe_gpu_axpy<Dtype>(channels_, Dtype(1), ones_data, bottom_diff);
    bottom_diff += channels_;
  }
}

INSTANTIATE_CLASS(ClassEntropyLossLayer);


}  // namespace caffe
