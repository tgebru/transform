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
void AccumLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  if (this->layer_param_.has_accum_param())
    discount_coeff_ = this->layer_param_.accum_param().discount_coeff();
  else
    discount_coeff_ = Dtype(1);
  num_ = bottom[0]->num();
  dim_ = bottom[0]->count() / num_;
  (*top)[0]->Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  ones_.Reshape(num_, 1, 1, 1);
  avg_blob_.ReshapeLike(*(*top)[0]);
  Dtype* avg_data = avg_blob_.mutable_cpu_data();
  Dtype* ones_data = ones_.mutable_cpu_data();
  for (int i=0; i<dim_; i++)
    avg_data[i] = Dtype(1) / Dtype(dim_); 
  for (int i=0; i<num_; i++)
    ones_data[i] = Dtype(1) / Dtype(num_);
  alpha_ = this->layer_param_.accum_param().init_alpha();
}

template <typename Dtype>
Dtype AccumLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  
  const Dtype* data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* ones_data = ones_.cpu_data();
  Dtype* avg_data = avg_blob_.mutable_cpu_data();
  
  alpha_ = alpha_ * discount_coeff_ + Dtype(1);
  caffe_cpu_gemv<Dtype>(CblasTrans, num_, dim_, Dtype(1) / alpha_, data, ones_data, (alpha_ - Dtype(1)) / alpha_, avg_data);
  caffe_copy<Dtype>(dim_, avg_data, top_data);
  
  return Dtype(0);
}

template <typename Dtype>
void AccumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  
  for (int i=0; i<num_; i++) {
    caffe_copy<Dtype>(dim_, top_diff, bottom_diff);    
    bottom_diff += dim_;
  }
  bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  caffe_scal<Dtype>(num_*dim_, Dtype(1) / alpha_ / num_, bottom_diff);
}


INSTANTIATE_CLASS(AccumLayer);


}  // namespace caffe
