// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
__global__ void im2perpixel_gpu_kernel(const int n, const Dtype* data_im,
    const int channels, const int height, const int width,
    Dtype* data_perpixel) {
  CUDA_KERNEL_LOOP(index, n) {
    int k = index;
    int w = k % width;
    k /= width;
    int h = k % height;
    k /= height;
    data_im += (k * channels * height + h) * width + w;
    data_perpixel += ((k * height + h) * width + w) * channels;
    for (int c = 0; c < channels; ++c) {
      *data_perpixel = *data_im;
      data_perpixel++;
      data_im += height * width; 
    }
  }
}

template <typename Dtype>
__global__ void perpixel2im_gpu_kernel(const int n, const Dtype* data_perpixel,
    const int channels, const int height, const int width,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    int k = index;
    int w = k % width;
    k /= width;
    int h = k % height;
    k /= height;
    data_im += (k * channels * height + h) * width + w;
    data_perpixel += ((k * height + h) * width + w) * channels;
    for (int c = 0; c < channels; ++c) {
      *data_im = *data_perpixel;
      data_perpixel++;
      data_im += height * width; 
    }
  }
}
  
template <typename Dtype>
Dtype Im2perPixelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int num_kernels = num_ * height_ * width_;
  im2perpixel_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, bottom_data, channels_, height_, width_, top_data);
  return Dtype(0.);
}

template <typename Dtype>
void Im2perPixelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  int num_kernels = num_ * height_ * width_;
  perpixel2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, top_diff, channels_, height_, width_, bottom_diff);
}


INSTANTIATE_CLASS(Im2perPixelLayer);

}  // namespace caffe
