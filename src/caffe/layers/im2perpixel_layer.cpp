// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
void Im2perPixelLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  (*top)[0]->Reshape(num_ * width_ * height_, channels_, 1, 1);
}

template <typename Dtype>
Dtype Im2perPixelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int ibot, itop;
  for (int n = 0; n < num_; ++n) {
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
        for (int c = 0; c < channels_; ++c) {
          ibot = ((n * channels_ + c) * height_ + h) * width_ + w;
          itop = ((n * height_ + h) * width_ + w) * channels_ + c;
          top_data[itop] = bottom_data[ibot];
        }
      }
    }
  }
  return Dtype(0.);
}

template <typename Dtype>
void Im2perPixelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  int ibot, itop;
  for (int n = 0; n < num_; ++n) {
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
        for (int c = 0; c < channels_; ++c) {
          ibot = ((n * channels_ + c) * height_ + h) * width_ + w;
          itop = ((n * height_ + h) * width_ + w) * channels_ + c;
          bottom_diff[ibot] = top_diff[itop];
        }
      }
    }
  }
}

INSTANTIATE_CLASS(Im2perPixelLayer);

}  // namespace caffe
