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
void ClassEntropyLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  discount_coeff_ = this->layer_param_.class_entropy_param().discount_coeff();
  if (this->layer_param_.has_loss_param())
    coeff_ = this->layer_param_.loss_param().coeff();
  else
    coeff_ = Dtype(1);
  channels_ = bottom[0]->channels();
  num_ = bottom[0]->num();
  class_prob_.Reshape(1, channels_, 1, 1);
  log_class_prob_.ReshapeLike(class_prob_);
  ones_.Reshape(1, channels_, 1, 1);
  avg_.Reshape(num_, 1, 1, 1);
  Dtype* class_prob_data = class_prob_.mutable_cpu_data();
  Dtype* avg_data = avg_.mutable_cpu_data();
  Dtype* ones_data = ones_.mutable_cpu_data();
  for (int i=0; i<channels_; i++) {
    class_prob_data[i] = Dtype(0);//Dtype(1.) / channels_;
    ones_data[i] = Dtype(1);
  }  
  for (int i=0; i<num_; i++)
    avg_data[i] = Dtype(1) / num_;
  alpha_ = Dtype(0);
}

template <typename Dtype>
Dtype ClassEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  
  const Dtype* data = bottom[0]->cpu_data();
  const Dtype* avg_data = avg_.cpu_data();
  Dtype* class_prob_data = class_prob_.mutable_cpu_data();
  Dtype* log_class_prob_data = log_class_prob_.mutable_cpu_data(); 
  
  alpha_ = alpha_ * discount_coeff_ + Dtype(1);
  caffe_cpu_gemv(CblasTrans, num_, channels_, Dtype(1) / alpha_, data, avg_data, (alpha_ - Dtype(1)) / alpha_, class_prob_data);
  
  for (int i=0; i<channels_; i++)
    LOG(INFO) << "class_prob_data[" << i <<"]=" << class_prob_data[i];
  
  for (int i=0; i<channels_; i++) 
    log_class_prob_data[i] = log(max(class_prob_data[i], Dtype(FLT_MIN)));
  
  Dtype loss = coeff_ * caffe_cpu_dot<Dtype>(channels_, log_class_prob_data, class_prob_data);
  
  LOG(INFO) << "loss=" << loss;
  
  return loss;
}

template <typename Dtype>
void ClassEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* log_class_prob_data = log_class_prob_.cpu_data();
  const Dtype* ones_data = ones_.cpu_data();
  
  for (int i=0; i<num_; i++) {
    caffe_copy<Dtype>(channels_, log_class_prob_data, bottom_diff);
    caffe_axpy<Dtype>(channels_, Dtype(1), ones_data, bottom_diff);
    bottom_diff += channels_;
  }
  
  bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  for (int i=0; i<(*bottom)[0]->count(); i++)
    LOG(INFO) << "bottom_diff[" << i << "]=" << bottom_diff[i];
}


INSTANTIATE_CLASS(ClassEntropyLossLayer);


}  // namespace caffe
