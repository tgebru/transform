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
void SoftmaxKLLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob1_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
  prob2_.ReshapeLike(prob1_);
  log_vector_.ReshapeLike(prob1_);
  if (this->layer_param_.has_loss_param())
    coeff_ = this->layer_param_.loss_param().coeff();
  else
    coeff_ = Dtype(1);
  ones_.Reshape(prob1_.channels(), prob1_.channels(), 1, 1);
  Dtype* ones_data = ones_.mutable_cpu_data();
  for (int i=0; i<ones_.count(); i++)
    ones_data[i] = Dtype(1);
  kl_persample_.ReshapeLike(prob1_);
}

template <typename Dtype>
Dtype SoftmaxKLLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_top_vec_[0] = &prob1_;
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  softmax_bottom_vec_[0] = bottom[1];
  softmax_top_vec_[0] = &prob2_;
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  
  const Dtype* prob1_data = prob1_.cpu_data();
  const Dtype* prob2_data = prob2_.cpu_data();
  Dtype* log_data = log_vector_.mutable_cpu_data();
  int num = prob1_.num();
  int count = prob1_.count();
  
  const Dtype* bottom1_data = bottom[0]->cpu_data();
  const Dtype* bottom2_data = bottom[1]->cpu_data();
  
  for (int i=0; i<count; i++) { 
    log_data[i] = (log(max(prob1_data[i], Dtype(kLOG_THRESHOLD))) - log(max(prob2_data[i], Dtype(kLOG_THRESHOLD))));
//     LOG(INFO) << "log_data[" << i << "]=" << log_data[i] << ", prob1_data[" << i << "]=" << prob1_data[i]
//       << ", prob2_data[" << i << "]=" << prob2_data[i]
//       << ", bottom1_data[" << i << "]=" << bottom1_data[i]
//       << ", bottom2_data[" << i << "]=" << bottom2_data[i];
  }
  loss_ = caffe_cpu_dot<Dtype>(count, log_data, prob1_data);
//   LOG(INFO) << "loss_=" << loss_ << ", coeff_=" << coeff_ << ". num=" << num;
  return loss_ * coeff_ / Dtype(num);
}

template <typename Dtype>
void SoftmaxKLLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff1 = (*bottom)[0]->mutable_cpu_diff();
  Dtype* bottom_diff2 = (*bottom)[1]->mutable_cpu_diff();
  const Dtype* prob1_data = prob1_.cpu_data();
  const Dtype* prob2_data = prob2_.cpu_data();
  const Dtype* log_data = log_vector_.cpu_data();
  Dtype* ones_data = ones_.mutable_cpu_data();
  Dtype* kl_persample_data = kl_persample_.mutable_cpu_data();
  int num = prob1_.num();
  int channels = prob1_.channels();
  int count = prob1_.count();
  
  caffe_mul<Dtype>(count, log_data, prob1_data, bottom_diff1);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels, channels, Dtype(1),
      bottom_diff1, ones_data, Dtype(0), kl_persample_data);
  caffe_mul<Dtype>(count, kl_persample_data, prob1_data, kl_persample_data);
  caffe_axpy<Dtype>(count, -1, kl_persample_data, bottom_diff1);  
  caffe_scal(count, coeff_ / Dtype(num), bottom_diff1);
  
  caffe_copy<Dtype>(count, prob2_data, bottom_diff2);
  caffe_axpy<Dtype>(count, Dtype(-1), prob1_data, bottom_diff2);  
  caffe_scal(count, coeff_ / Dtype(num), bottom_diff2);

}


INSTANTIATE_CLASS(SoftmaxKLLossLayer);


}  // namespace caffe
