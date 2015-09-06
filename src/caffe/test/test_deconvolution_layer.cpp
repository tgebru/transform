// Copyright 2014 BVLC and contributors.

#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class DeConvolutionLayerTest : public ::testing::Test {
 protected:
  DeConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_deconv_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_bottom_->Reshape(2, 3, 6, 4);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_deconv_vec_.push_back(blob_deconv_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~DeConvolutionLayerTest() { delete blob_bottom_; delete blob_top_; delete blob_deconv_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_deconv_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_deconv_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(DeConvolutionLayerTest, Dtypes);

TYPED_TEST(DeConvolutionLayerTest, TestCPUSimpleConvolution) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::CPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 27.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], 27.1, 1e-4);
  }

  // test the deconvolution
  DeConvolutionParameter* deconvolution_param =
      layer_param.mutable_deconvolution_param();
  deconvolution_param->set_kernel_size(3);
  deconvolution_param->set_stride(2);
  deconvolution_param->set_output_channels(3);
  deconvolution_param->set_output_height(6);
  deconvolution_param->set_output_width(4);
  deconvolution_param->mutable_weight_filler()->set_type("constant");
  deconvolution_param->mutable_weight_filler()->set_value(1);
  deconvolution_param->mutable_bias_filler()->set_type("constant");
  deconvolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > dlayer(new DeConvolutionLayer<TypeParam>(layer_param));
  dlayer->SetUp(this->blob_top_vec_, &(this->blob_deconv_vec_));
  dlayer->Forward(this->blob_top_vec_, &(this->blob_deconv_vec_));

  /* JTS debug
  cout << this->blob_top_->num() << " " << this->blob_top_->channels() << " " << this->blob_top_->height() << " " << this->blob_top_->width() << endl;
  cout << this->blob_bottom_->num() << " " << this->blob_bottom_->channels() << " " << this->blob_bottom_->height() << " " << this->blob_bottom_->width() << endl;
  cout << this->blob_deconv_->num() << " " << this->blob_deconv_->channels() << " " << this->blob_deconv_->height() << " " << this->blob_deconv_->width() << endl;
  */
  // using backward on the normal conv layer should be equivalent to deconv forward (except for the bias)! 
  for (int i = 0; i < this->blob_top_->count(); ++i) {
      this->blob_top_->mutable_cpu_diff()[i] = this->blob_top_->cpu_data()[i];
  }
  layer->Backward(this->blob_top_vec_, true, &(this->blob_bottom_vec_));
  //cout << " ddata ";
  for (int i = 0; i < this->blob_deconv_->count(); ++i) {
      EXPECT_NEAR(this->blob_bottom_->cpu_diff()[i], this->blob_deconv_->cpu_data()[i], 0.2 + 1e-4);
      //cout << this->blob_deconv_->cpu_data()[i] << " ";
  }
  //cout << endl;

  // next test deconv backward pass
  // it should be equivalent to conv forward pass up to a constant (the bias)
  // setup diff to be equivalent to bottom for testing this
  for (int i = 0; i < this->blob_deconv_->count(); ++i) {
      this->blob_deconv_->mutable_cpu_diff()[i] = this->blob_bottom_->cpu_data()[i];
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
      this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  // next run the backward pass
  dlayer->Backward(this->blob_deconv_vec_, true, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(this->blob_top_->cpu_diff()[i] , 27, 1e-4);
      //cout << this->blob_top_->cpu_diff()[i] << " ";
  }
  // the gradient wrt. weights should be 1. * deconv_vec (which is 108.4)
}

TYPED_TEST(DeConvolutionLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  DeConvolutionParameter* deconvolution_param =
      layer_param.mutable_deconvolution_param();
  deconvolution_param->set_kernel_size(3);
  deconvolution_param->set_stride(2);
  deconvolution_param->set_output_channels(3);
  deconvolution_param->set_output_height(6);
  deconvolution_param->set_output_width(4);
  deconvolution_param->mutable_weight_filler()->set_type("gaussian");
  deconvolution_param->mutable_bias_filler()->set_type("gaussian");
  Caffe::set_mode(Caffe::CPU);
  DeConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(DeConvolutionLayerTest, TestGPUGradient) {
  LayerParameter layer_param;
  DeConvolutionParameter* deconvolution_param =
      layer_param.mutable_deconvolution_param();
  deconvolution_param->set_kernel_size(3);
  deconvolution_param->set_stride(2);
  deconvolution_param->set_output_channels(3);
  deconvolution_param->set_output_height(6);
  deconvolution_param->set_output_width(4);
  deconvolution_param->mutable_weight_filler()->set_type("gaussian");
  deconvolution_param->mutable_bias_filler()->set_type("gaussian");
  Caffe::set_mode(Caffe::GPU);
  DeConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

}  // namespace caffe
