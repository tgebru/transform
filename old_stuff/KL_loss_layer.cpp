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
void SoftmaxKLLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  log_vector_.ReshapeLike(*(bottom[0]));
  if (this->layer_param_.has_loss_param())
    coeff_ = this->layer_param_.loss_param().coeff();
  else
    coeff_ = Dtype(1);
}

template <typename Dtype>
Dtype SoftmaxKLLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  
  Dtype* log_data = log_vector_.mutable_cpu_data();
  const Dtype* prob1_data = bottom[0]->cpu_data();
  const Dtype* prob2_data = bottom[1]->cpu_data();
  
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  
  for (int i=0; i<count; i++) { 
    log_data[i] = (log(max(prob1_data[i], Dtype(FLT_MIN))) - log(max(prob2_data[i], Dtype(FLT_MIN))));
  }
  loss_ = coeff_ / num * caffe_cpu_dot<Dtype>(count, log_data, prob1_data);

  return loss_;
}

template <typename Dtype>
void SoftmaxKLLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff1 = (*bottom)[0]->mutable_cpu_diff();
  Dtype* bottom_diff2 = (*bottom)[1]->mutable_cpu_diff();
  const Dtype* prob1_data = (*bottom)[0]->cpu_data();
  const Dtype* prob2_data = (*bottom)[1]->cpu_data();
  const Dtype* log_data = log_vector_.cpu_data();
  
  int num = (*bottom)[0]->num();
  int count = (*bottom)[0]->count();  
  
  caffe_copy<Dtype>(count, prob2_data, bottom_diff2);
  caffe_axpy<Dtype>(count, Dtype(-1), prob1_data, bottom_diff2);  
  caffe_scal(count, coeff_ / num, bottom_diff2);
  
  caffe_mul<Dtype>(count, log_data, prob1_data, bottom_diff1);
  caffe_axpy<Dtype>(count, -loss_, prob1_data, bottom_diff1);  
  caffe_scal(count, coeff_ / num, bottom_diff1);

}


INSTANTIATE_CLASS(SoftmaxKLLossLayer);


}  // namespace caffe
