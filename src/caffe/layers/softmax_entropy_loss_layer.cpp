// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxEntropyLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
  log_vector_.ReshapeLike(prob_);
  if (this->layer_param_.has_loss_param())
    coeff_ = this->layer_param_.loss_param().coeff();
  else
    coeff_ = Dtype(1);
  if (coeff_ < 0) 
    min_val_ = log(Dtype(bottom[0]->count() / bottom[0]->num())) * coeff_;
  else
    min_val_ = Dtype(0);
  ones_.Reshape(prob_.channels(), prob_.channels(), 1, 1);
  Dtype* ones_data = ones_.mutable_cpu_data();
  for (int i=0; i<ones_.count(); i++)
    ones_data[i] = Dtype(1);
  e_persample_.ReshapeLike(prob_);
}

template <typename Dtype>
Dtype SoftmaxEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_top_vec_[0] = &prob_;
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  
  const Dtype* prob_data = prob_.cpu_data();
  Dtype* log_data = log_vector_.mutable_cpu_data();
  int num = prob_.num();
  int count = prob_.count();
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  
  for (int i=0; i<count; i++) { 
    log_data[i] = log(max(prob_data[i], Dtype(kLOG_THRESHOLD)));
  }
  loss_ = caffe_cpu_dot<Dtype>(count, log_data, prob_data);
//   LOG(INFO) << "loss_=" << loss_ << ", coeff_=" << coeff_ << ". num=" << num;
  return - min_val_ - loss_ * coeff_ / Dtype(num);
}

template <typename Dtype>
void SoftmaxEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* log_data = log_vector_.cpu_data();
  Dtype* ones_data = ones_.mutable_cpu_data();
  Dtype* e_persample_data = e_persample_.mutable_cpu_data();
  int num = prob_.num();
  int channels = prob_.channels();
  int count = prob_.count();
  
  caffe_mul<Dtype>(count, log_data, prob_data, bottom_diff);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels, channels, Dtype(1),
      bottom_diff, ones_data, Dtype(0), e_persample_data);
  caffe_mul<Dtype>(count, e_persample_data, prob_data, e_persample_data);
  caffe_axpy<Dtype>(count, -1, e_persample_data, bottom_diff);  
  caffe_scal(count, -coeff_ / Dtype(num), bottom_diff);

}


INSTANTIATE_CLASS(SoftmaxEntropyLossLayer);


}  // namespace caffe
