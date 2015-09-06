// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
  
template <typename Dtype>
__global__ void gpu_log(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = log(max(in[index], Dtype(kLOG_THRESHOLD)));
  }
}

template <typename Dtype>
Dtype EntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    
  const Dtype* data = bottom[0]->gpu_data();
  Dtype* log_data = log_blob_.mutable_gpu_data(); 
  
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  
  gpu_log<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, data, log_data);
  
  Dtype loss;
  caffe_gpu_scal<Dtype>(count, -coeff_, log_data);
  caffe_gpu_dot<Dtype>(count, log_data, data, &loss);
  
  return loss / Dtype(num) - min_val_;
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const Dtype* log_data = log_blob_.gpu_data();
  
  int count = (*bottom)[0]->count();
  int num = (*bottom)[0]->num();
  
  caffe_gpu_copy<Dtype>(count, log_data, bottom_diff);
  caffe_gpu_add_scalar<Dtype>(count, -coeff_, bottom_diff);
  caffe_gpu_scal<Dtype>(count, Dtype(1)/Dtype(num), bottom_diff);
}

INSTANTIATE_CLASS(EntropyLossLayer);


}  // namespace caffe
