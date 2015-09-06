// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void SubspaceMaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_channels,
    const int kernel_size, const int stride, const int pad, Dtype* top_data,
    int* mask, Dtype* top_mask) {
    // JTS Note: the following implementation is very naive 
    //           and will result in a lot of non coalescent memory access
    //           it could be made more efficient by
    //           temporarily copying data to a buffer first and then computing
    //           but for a quick implementation this will do!
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int pc = (index / width / height) % pooled_channels;
    int n = index / width / height / pooled_channels;
    int cstart = pc * stride;
    int cend = min(cstart + kernel_size, channels);
    cstart = max(cstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    for (int c = cstart; c < cend; ++c) {
        const int idx = ((n * channels + c) * height  + h) * width + w;
        if (bottom_data[idx] > maxval) {
            maxidx = idx;
            maxval = bottom_data[idx];
        }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void SubspaceAvePoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_channels,
    const int kernel_size, const int stride, const int pad, Dtype* top_data) {
    // JTS Note: the following implementation is very naive 
    //           and will result in a lot of non coalescent memory access
    //           it could be made more efficient by
    //           temporarily copying data to a buffer first and then computing
    //           but for a quick implementation this will do!
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int pc = (index / width / height) % pooled_channels;
    int n = index / width / height / pooled_channels;
    int cstart = pc * stride;
    int cend = min(cstart + kernel_size, channels);
    int pool_size = (cend - cstart);
    cstart = max(cstart, 0);
    Dtype aveval = 0;
    for (int c = cstart; c < cend; ++c) {
        const int idx = ((n * channels + c) * height  + h) * width + w;
        aveval += bottom_data[idx];
    }
    top_data[index] = aveval / pool_size;
  }
}

template <typename Dtype>
Dtype SubspacePoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top->size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = (*top)[1]->mutable_gpu_data();
    } else {
      mask = max_idx_->mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    SubspaceMaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_channels_, kernel_size_, stride_,
        pad_, top_data, mask, top_mask);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SubspaceAvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_channels_, kernel_size_, stride_,
        pad_, top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
  return Dtype(0.);
}


template <typename Dtype>
__global__ void SubspaceMaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_channels,
    const int kernel_size, const int stride,
    const int pad, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int pcstart = (c + pad < kernel_size) ? 0 : (c + pad - kernel_size) / stride + 1;
    int pcend = min((c + pad) / stride + 1, pooled_channels);
    Dtype gradient = 0;
    if (mask) {
      for (int pc = pcstart; pc < pcend; ++pc) {
          const int top_index = ((n * pooled_channels + pc) * height  + h) * width + w;
          if (mask[top_index] == index) {
              gradient += top_diff[top_index];
          }
      }
    } else {
      for (int pc = pcstart; pc < pcend; ++pc) {
        const int top_index = ((n * pooled_channels + pc) * height  + h) * width + w;
        if (top_mask[top_index] == index) {
            gradient += top_diff[top_index];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void SubspaceAvePoolBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_channels,
    const int kernel_size, const int stride, const int pad,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width + pad;
    int h = (index / width) % height + pad;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int pcstart = (c + pad < kernel_size) ? 0 : (c + pad - kernel_size) / stride + 1;
    int pcend = min((c + pad) / stride + 1, pooled_channels);
    Dtype gradient = 0;
    for (int pc = pcstart; pc < pcend; ++pc) {
        const int cstart = pc * stride;
        const int cend = min(cstart + kernel_size, channels);
        const int pool_size = (cend - cstart);
        const int top_index = ((n * pooled_channels + pc) * height  + h) * width + w;
        gradient += top_diff[top_index] / pool_size;
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void SubspacePoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const int count = (*bottom)[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_->gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    SubspaceMaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_channels_,
        kernel_size_, stride_, pad_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SubspaceAvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_channels_, kernel_size_, stride_,
        pad_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_CLASS(SubspacePoolingLayer);


}  // namespace caffe
