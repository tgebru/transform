// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void UnPoolingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  max_top_blobs_ = 1;
  Layer<Dtype>::SetUp(bottom, top);
  kernel_size_ = this->layer_param_.pooling_param().kernel_size();
  stride_ = this->layer_param_.pooling_param().stride();
  pad_ = this->layer_param_.pooling_param().pad();
  if (pad_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_, kernel_size_);
  }
  channels_ = bottom[0]->channels();
  pooled_height_ = bottom[0]->height();
  pooled_width_ = bottom[0]->width();
  //pooled_height_ = static_cast<int>(ceil(static_cast<float>(
  //    height_ + 2 * pad_ - kernel_size_) / stride_)) + 1;
  //pooled_width_ = static_cast<int>(ceil(static_cast<float>(
  //    width_ + 2 * pad_ - kernel_size_) / stride_)) + 1;
  height_ = static_cast<int>(ceil(static_cast<float>((pooled_height_ - 1) * stride_))) - 2 * pad_ + kernel_size_;
  width_ = static_cast<int>(ceil(static_cast<float>((pooled_width_ - 1) * stride_))) - 2 * pad_ + kernel_size_;
  if (pad_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last    // JTS check
    //if ((pooled_height_ - 1) * stride_ >= height_ + pad_) {
    //  --pooled_height_;
    //}
    //if ((pooled_width_ - 1) * stride_ >= width_ + pad_) {
    //  --pooled_width_;
    //}
    CHECK_LT((pooled_height_ - 1) * stride_, height_ + pad_);
    CHECK_LT((pooled_width_ - 1) * stride_, width_ + pad_);
  }
  (*top)[0]->Reshape(bottom[1]->num(), channels_, height_,
                     width_);
}



template <typename Dtype>
Dtype UnPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  //JTS DONE -> const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  //JTS DONE ->  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  caffe_set((*top)[0]->count(), Dtype(0), top_data);
  // JTS done -> const Dtype* top_mask;
  const Dtype* bottom_mask;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // We require a bottom mask in position 1!
    assert(bottom.size() > 1);
    // The main loop
    bottom_mask = bottom[1]->cpu_data();
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index = bottom_mask[index];
            top_data[bottom_index] += bottom_data[index];
            //std::cout << "midx = " << bottom_index << " val: " << bottom_data[index] << " == " << top_data[bottom_index] << std::endl;
          }
        }
        top_data += (*top)[0]->offset(0, 1);
        bottom_data += bottom[0]->offset(0, 1);

        bottom_mask += bottom[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_ - pad_;
            int wstart = pw * stride_ - pad_;
            int hend = min(hstart + kernel_size_, height_ + pad_);
            int wend = min(wstart + kernel_size_, width_ + pad_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[h * width_ + w] +=
                  bottom_data[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        top_data += (*top)[0]->offset(0, 1);
        bottom_data += bottom[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  return Dtype(0.);
}


template <typename Dtype>
void UnPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down) {
    return;
  }

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);

  const Dtype* top_mask;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    assert(bottom->size() > 1);
    top_mask = (*bottom)[1]->cpu_data();
    for (int n = 0; n < (*bottom)[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index = top_mask[index];
            bottom_diff[index] += top_diff[bottom_index];
            //std::cout << "midx = " << bottom_index << " val: " << top_diff[bottom_index] << " == " << bottom_diff[index] << std::endl;
          }
        }
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        top_mask += (*bottom)[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
      for (int n = 0; n < (*bottom)[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_ - pad_;
            int wstart = pw * stride_ - pad_;
            int hend = min(hstart + kernel_size_, height_ + pad_);
            int wend = min(wstart + kernel_size_, width_ + pad_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[ph * pooled_width_ + pw] +=
                  top_diff[h * width_ + w];
              }
            }
            bottom_diff[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // offset
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


INSTANTIATE_CLASS(UnPoolingLayer);


}  // namespace caffe
