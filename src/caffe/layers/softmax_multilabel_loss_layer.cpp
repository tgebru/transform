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
void SoftmaxMultilabelLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
  log_prob_.ReshapeLike(prob_);
  log_label_.ReshapeLike(prob_);
}

template <typename Dtype>
Dtype SoftmaxMultilabelLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  Dtype* log_prob_data = log_prob_.mutable_cpu_data();
  Dtype* log_label_data = log_label_.mutable_cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int count = prob_.count();
  int dim =  count / num;
  Dtype loss;
  for (int i=0; i<count; i++) { 
    log_prob_data[i] = -log(max(prob_data[i], Dtype(kLOG_THRESHOLD)));
    log_label_data[i] = -log(max(label[i], Dtype(kLOG_THRESHOLD)));
  }  
  loss = caffe_cpu_dot<Dtype>(count, label, log_prob_data);
//   loss -= caffe_cpu_dot<Dtype>(count, label, log_label_data);    // comment out to make gradient tests work
  return loss / num;
}

template <typename Dtype>
void SoftmaxMultilabelLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff_net = (*bottom)[0]->mutable_cpu_diff();
  Dtype* bottom_diff_lbl = (*bottom)[1]->mutable_cpu_diff();
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* log_prob_data = log_prob_.cpu_data();
  int num = prob_.num();
  int count = prob_.count();
  int dim = count / num;  
  
  memcpy(bottom_diff_net, prob_data, sizeof(Dtype) * count);
  const Dtype* label = (*bottom)[1]->cpu_data();  
  caffe_axpy<Dtype>(count, Dtype(-1), label, bottom_diff_net);
  // Scale down gradient
  caffe_scal(count, Dtype(1) / num, bottom_diff_net);
  
  memcpy(bottom_diff_lbl, log_prob_data, sizeof(Dtype) * count);
  caffe_scal(count, Dtype(1) / num, bottom_diff_lbl);
}


INSTANTIATE_CLASS(SoftmaxMultilabelLossLayer);


}  // namespace caffe
