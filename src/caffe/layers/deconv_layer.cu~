// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype DeConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  //const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();

  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();

  
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int bottom_offset = M_ * N_;
  for (int n = 0; n < num_; ++n) {
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + weight_offset * g,
          bottom_data + bottom[0]->offset(n) + bottom_offset * g,
          (Dtype)0., col_data + col_offset * g);
        // bias - TODO
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, K_, 1,
                              (Dtype)1., this->blobs_[1]->gpu_data(),
                              reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
                              (Dtype)1., col_data + col_offset * g);
      }
      // col2im forward to the top_data
      col2im_gpu(col_data, channels_, height_out_, width_out_, kernel_size_, pad_,
                 stride_, top_data + (*top)[0]->offset(n));
  }
  /* Debugging stuff
  for (int n = 0; n < col_buffer_.count(); ++n) {
      std::cout << col_buffer_.cpu_data()[n] <<  "  "; 
  }
  std::cout << std::endl;
  */
  return Dtype(0.);
}

template <typename Dtype>
void DeConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  Dtype* bias_diff = NULL;
  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    CUDA_CHECK(cudaMemset(bias_diff, 0,
        sizeof(Dtype) * this->blobs_[1]->count()));
  }

  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int bottom_offset = M_ * N_;
  CUDA_CHECK(cudaMemset(weight_diff, 0,
      sizeof(Dtype) * this->blobs_[0]->count()));
  for (int n = 0; n < num_; ++n) {
      im2col_gpu(top_diff + top[0]->offset(n), channels_, height_out_,
                 width_out_, kernel_size_, pad_, stride_, col_diff);
      // gradient wrt. weights
      for (int g = 0; g < group_; ++g) {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                                (Dtype)1., bottom_data + (*bottom)[0]->offset(n) + bottom_offset * g,
                                col_diff + col_offset * g, (Dtype)1.,
                                weight_diff + weight_offset * g);
          // gradient wrt. bias
          if (bias_term_) {
              caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_,
                                    (Dtype)1., reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
                                    col_diff + col_offset * g, (Dtype)1.,
                                    bias_diff);
          }
      }
      if (propagate_down) {
          for (int g = 0; g < group_; ++g) {
              caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
                                    (Dtype)1., weight + weight_offset * g, col_diff + col_offset * g,
                                    (Dtype)0., bottom_diff + (*bottom)[0]->offset(n) + bottom_offset * g);
          }
      }
  }
  /* debug
  for (int n = 0; n < this->blobs_[0]->count(); ++n) {
      std::cout << this->blobs_[0]->cpu_diff()[n] << std::endl;
  }
  for (int n = 0; n < col_buffer_.count(); ++n) {
      //std::cout << col_buffer_.cpu_diff()[n] <<  "  "; 
      std::cout << top[0]->cpu_diff()[n] <<  "  "; 
  }
  std::cout << std::endl;
  */
}


INSTANTIATE_CLASS(DeConvolutionLayer);

}  // namespace caffe
