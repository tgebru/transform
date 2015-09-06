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
void SubspacePoolingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Equivalent to the the normal Pooling layer we need to set
  // the max number of top blobs before calling base Layer::SetUp.
  // If doing MAX pooling, we can optionally output an extra top Blob
  // for the mask.  Otherwise, we only have one top Blob.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX) {
    max_top_blobs_ = 2;
  } else {
    max_top_blobs_ = 1;
  }
  Layer<Dtype>::SetUp(bottom, top);
  kernel_size_ = this->layer_param_.pooling_param().kernel_size();
  stride_ = this->layer_param_.pooling_param().stride();
  // contrary to normal pooling we do not support padding here right now
  // i.e. the number of channels always has to be divisible by the kernel size
  pad_ = this->layer_param_.pooling_param().pad();
  CHECK(pad_ == false) << "Padding not implemented for subspace pooling.";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  pooled_channels_ = static_cast<int>(ceil(static_cast<float>(
      channels_ - kernel_size_) / stride_)) + 1;

  (*top)[0]->Reshape(bottom[0]->num(), pooled_channels_, height_,
                     width_);
  if (top->size() > 1) {
    (*top)[1]->ReshapeLike(*(*top)[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top->size() == 1) {
    max_idx_.reset(new Blob<int>(bottom[0]->num(), pooled_channels_,
                                 height_, width_));
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), pooled_channels_, height_,
                      width_);
  }
}

template <typename Dtype>
Dtype SubspacePoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int top_count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top->size() > 1;
  int* mask;
  Dtype* top_mask;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = (*top)[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_->mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int pc = 0; pc < pooled_channels_; ++pc) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int cstart = pc * stride_;
            int cend = min(cstart + kernel_size_, channels_);
            cstart = max(cstart, 0);
            const int pool_index = (*top)[0]->offset(n, pc, h, w);
            for (int c = cstart; c < cend; ++c) {
                const int index = bottom[0]->offset(n, c, h, w);
                if (bottom_data[index] > top_data[pool_index]) {
                    top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                      top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                      mask[pool_index] = index;
                  }
                }
            }
          }
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int pc = 0; pc < pooled_channels_; ++pc) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int cstart = pc * stride_;
            int cend = min(cstart + kernel_size_, channels_);
            int pool_size = (cend - cstart);
            cstart = max(cstart, 0);
            const int pool_index = (*top)[0]->offset(n, pc, h, w);
            for (int c = cstart; c < cend; ++c) {
                const int index = bottom[0]->offset(n, c, h, w);
                top_data[pool_index] +=
                    bottom_data[index];
            }
            top_data[pool_index] /= pool_size;
          }
        }
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
void SubspacePoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
  // We'll use the mask from to top[1] if it is provided
  const bool use_top_mask = top.size() > 1;
  const int* mask;
  const Dtype* top_mask;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_->cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int pc = 0; pc < pooled_channels_; ++pc) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            const int index = top[0]->offset(n, pc, h, w);
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int pc = 0; pc < pooled_channels_; ++pc) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int cstart = pc * stride_;
            int cend = min(cstart + kernel_size_, channels_);
            int pool_size = (cend - cstart);
            cstart = max(cstart, 0);
            const int index = top[0]->offset(n, pc, h, w);
            for (int c = cstart; c < cend; ++c) {
                const int bottom_index = (*bottom)[0]->offset(n, c, h, w);
                bottom_diff[bottom_index] += top_diff[index] / pool_size;
            }
          }
        }
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


INSTANTIATE_CLASS(SubspacePoolingLayer);


}  // namespace caffe
