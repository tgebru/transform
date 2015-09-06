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
class NormConstraintTest : public ::testing::Test {
 protected:
    NormConstraintTest()
      : blob_bottom_(new Blob<Dtype>(5, 3, 28, 28)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~NormConstraintTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(NormConstraintTest, Dtypes);

TYPED_TEST(NormConstraintTest, TestNorm) {
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  Caffe::set_mode(Caffe::CPU);
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  shared_ptr<InnerProductLayer<TypeParam> > layer(
      new InnerProductLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 10);
  TypeParam *weight = layer->blobs()[0]->mutable_cpu_data();
  int M = layer->blobs()[0]->height();
  int N = layer->blobs()[0]->width();
  // compute l2 norm
  for (int i = 0; i < N; ++i) {      
      TypeParam nrm = caffe_cpu_norm2(M, weight, N);
      // and make sure it is correctly computed
      TypeParam nrm_base = 0;
      for (int j = 0; j < M; ++j) {
          TypeParam data = layer->blobs()[0]->data_at(0, 0, j, i);
          nrm_base += data * data;
      }
      nrm_base = sqrt(nrm_base);
      EXPECT_NEAR(nrm, nrm_base, 1e-4);
      weight += layer->blobs()[0]->offset(0, 0, 0, 1);
  }

}

TYPED_TEST(NormConstraintTest, TestConstraintIP) {
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  Caffe::set_mode(Caffe::CPU);
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  shared_ptr<InnerProductLayer<TypeParam> > layer(
      new InnerProductLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 10);
  int M = layer->blobs()[0]->height();
  int N = layer->blobs()[0]->width();
  // apply the constraint 
  TypeParam mnorm = 1.;
  layer->normalize_weights(mnorm);
  // make sure all weight vectors now have norm 1
  for (int i = 0; i < N; ++i) {      
      TypeParam nrm_base = 0;
      for (int j = 0; j < M; ++j) {
          TypeParam data = layer->blobs()[0]->data_at(0, 0, j, i);
          nrm_base += data * data;
      }
      nrm_base = sqrt(nrm_base);
      EXPECT_LE(nrm_base, TypeParam(1) + 1e-3);
  }

}

TYPED_TEST(NormConstraintTest, TestConstraintIPGPU) {
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  Caffe::set_mode(Caffe::GPU);
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->mutable_weight_filler()->set_min(0.01);
  inner_product_param->mutable_bias_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  shared_ptr<InnerProductLayer<TypeParam> > layer(
      new InnerProductLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 10);
  int M = layer->blobs()[0]->height();
  int N = layer->blobs()[0]->width();
  // apply the constraint 
  TypeParam mnorm = 1.;
  layer->normalize_weights(mnorm);
  // make sure all weight vectors now have norm 1
  for (int i = 0; i < N; ++i) {      
      TypeParam nrm_base = 0;
      for (int j = 0; j < M; ++j) {
          TypeParam data = layer->blobs()[0]->data_at(0, 0, j, i);
          nrm_base += data * data;
      }
      nrm_base = sqrt(nrm_base);
      EXPECT_LE(nrm_base, TypeParam(1) + 1e-3);
  }

}


TYPED_TEST(NormConstraintTest, TestConstraintConvCPU) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("uniform");
  convolution_param->mutable_weight_filler()->set_min(0.);
  convolution_param->mutable_weight_filler()->set_max(1.);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::CPU);
  layer->normalize_weights(1.);
  // make sure all kernels now approximately have norm 1
  int N = layer->blobs()[0]->num();
  for (int i = 0; i < N; ++i) {      
      TypeParam nrm_base = 0;
      for (int j = 0; j < layer->blobs()[0]->channels(); ++j) {
          for (int k = 0; k < layer->blobs()[0]->height(); ++k) {
              for (int l = 0; l < layer->blobs()[0]->width(); ++l) {
                  TypeParam data = layer->blobs()[0]->data_at(i, j, k, l);
                  nrm_base += data * data;
              }
          }
      }
      nrm_base = sqrt(nrm_base);
      EXPECT_LE(nrm_base, TypeParam(1) + 1e-3);
  }
}

TYPED_TEST(NormConstraintTest, TestConstraintConvGPU) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("uniform");
  convolution_param->mutable_weight_filler()->set_min(0.);
  convolution_param->mutable_weight_filler()->set_max(1.);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffe::set_mode(Caffe::GPU);
  layer->normalize_weights(1.);
  // make sure all kernels now approximately have norm 1
  int N = layer->blobs()[0]->num();
  for (int i = 0; i < N; ++i) {      
      TypeParam nrm_base = 0;
      for (int j = 0; j < layer->blobs()[0]->channels(); ++j) {
          for (int k = 0; k < layer->blobs()[0]->height(); ++k) {
              for (int l = 0; l < layer->blobs()[0]->width(); ++l) {
                  TypeParam data = layer->blobs()[0]->data_at(i, j, k, l);
                  nrm_base += data * data;
              }
          }
      }
      nrm_base = sqrt(nrm_base);
      EXPECT_LE(nrm_base, TypeParam(1) + 1e-3);
  }
}

}  // namespace caffe
