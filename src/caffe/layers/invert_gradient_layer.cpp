// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InvertGradientLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  (*top)[0]->ReshapeLike(*(bottom[0]));
  gamma_ = this->layer_param_.coeff_schedule_param().gamma();
  initial_coeff_ = this->layer_param_.coeff_schedule_param().initial_coeff();
  final_coeff_ = this->layer_param_.coeff_schedule_param().final_coeff();
  iter_ = 0;
  coeff_ = initial_coeff_;
}

template <typename Dtype>
Dtype InvertGradientLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->ShareData(*bottom[0]);
  return Dtype(0.);
}

template <typename Dtype>
void InvertGradientLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  int count = top[0]->count();
  caffe_copy(count, top_diff, bottom_diff);
  caffe_scal(count, -coeff_, bottom_diff);
  iter_++;
  coeff_ = initial_coeff_ + (final_coeff_ - initial_coeff_) * (Dtype(2) / (Dtype(1) + exp(-gamma_ * iter_)) - Dtype(1));
}

INSTANTIATE_CLASS(InvertGradientLayer);

}  // namespace caffe
