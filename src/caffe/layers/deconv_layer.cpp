// Copyright 2014 BVLC and contributors.

#include <vector>

#include <iostream>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DeConvolutionLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  // we need the output of the corresponding deconvolution in order to determine the size
  kernel_size_ = this->layer_param_.deconvolution_param().kernel_size();
  stride_ = this->layer_param_.deconvolution_param().stride();
  group_ = this->layer_param_.deconvolution_param().group();
  pad_ = this->layer_param_.deconvolution_param().pad();
  height_out_ = this->layer_param_.deconvolution_param().output_height();
  width_out_ = this->layer_param_.deconvolution_param().output_width();
  // JTS TODO check num_ <-> num_output
  num_ = bottom[0]->num();
  // TODO read channels_ and num_output_ from optional second input
  channels_ = this->layer_param_.deconvolution_param().output_channels();
  //channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  //num_output_ = this->layer_param_.deconvolution_param().num_output();
  num_output_ = this->layer_param_.deconvolution_param().output_channels();
  int inverse_num_out = bottom[0]->channels();
  CHECK_GT(inverse_num_out, 0);
  CHECK_EQ(channels_ % group_, 0);
  // init im2col result buffer
  //std::cout << height_ << " " << width_ << std::endl;
  //std::cout << "nout "<< num_output_ << " " << height_out_ << " " << width_out_ << std::endl;
  col_buffer_.Reshape(
      1, channels_ * kernel_size_ * kernel_size_, height_, width_);
  // Set the parameters
  CHECK_EQ(inverse_num_out % group_, 0)
      << "Number of output should be multiples of group.";
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  // Figure out the dimensions for individual gemms.
  // JTS check N_ + K_
  M_ = inverse_num_out / group_;
  K_ = channels_ * kernel_size_ * kernel_size_ / group_;
  N_ = height_ * width_;
  (*top)[0]->Reshape(bottom[0]->num(), num_output_, height_out_, width_out_);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(
        inverse_num_out, channels_ / group_, kernel_size_, kernel_size_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.deconvolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    //std::cout << "wcount " << this->blobs_[0]->count() << std::endl;
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.deconvolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Set up the bias filler
  if (bias_term_) {
    bias_multiplier_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < N_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
}

template <typename Dtype>
Dtype DeConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  //const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  //const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  Dtype* col_data = col_buffer_.mutable_cpu_data();
  Dtype* col_diff = col_buffer_.mutable_cpu_diff();

  
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int bottom_offset = M_ * N_;
  for (int n = 0; n < num_; ++n) {
      for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + weight_offset * g,
          bottom_data + bottom[0]->offset(n) + bottom_offset * g,
          (Dtype)0., col_data + col_offset * g);
      }
      // col2im forward to the top_data
      col2im_cpu(col_data, channels_, height_out_, width_out_, kernel_size_, pad_,
                 stride_, top_data + (*top)[0]->offset(n));
      // add bias
      if (bias_term_) {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                                N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
                                reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
                                (Dtype)1., top_data + (*top)[0]->offset(n));
      }
      // Debugging stuff
      /*
      for (int k = 0; k < col_buffer_.count(); ++k) {
          std::cout << col_buffer_.cpu_data()[k] <<  "  "; 
      }
      std::cout << std::endl;
      */
  }
  
  return Dtype(0.);
}

template <typename Dtype>
void DeConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  Dtype* col_diff = col_buffer_.mutable_cpu_diff();
  Dtype* bias_diff = NULL;
  if (bias_term_) {
      bias_diff = this->blobs_[1]->mutable_cpu_diff();
      memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
      //JTS fixed gradient wrt. bias, not sure about the group stuff ...
      for (int n = 0; n < num_; ++n) {
            caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
                            1., top_diff + top[0]->offset(n), 
                            reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), 1.,
                            bias_diff);
      }
  }

  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int bottom_offset = M_ * N_;
  memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
  for (int n = 0; n < num_; ++n) {
      im2col_cpu(top_diff + top[0]->offset(n), channels_, height_out_,
                 width_out_, kernel_size_, pad_, stride_, col_diff);
      // gradient wrt. weights
      for (int g = 0; g < group_; ++g) {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                                (Dtype)1., bottom_data + (*bottom)[0]->offset(n) + bottom_offset * g,
                                col_diff + col_offset * g, (Dtype)1.,
                                weight_diff + weight_offset * g);
      }
      if (propagate_down) {
          for (int g = 0; g < group_; ++g) {
              caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
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

