// Copyright 2014 BVLC and contributors.

#include <cmath>
#include <cstdlib>
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
class SoftmaxKLLossLayerTest : public ::testing::Test {
 protected:
  SoftmaxKLLossLayerTest()
      : blob_bottom_data1_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_data2_(new Blob<Dtype>(10, 5, 1, 1)) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(5);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data1_);
    blob_bottom_vec_.push_back(blob_bottom_data1_);
    filler.Fill(this->blob_bottom_data2_);
    blob_bottom_vec_.push_back(blob_bottom_data2_);
  }
  virtual ~SoftmaxKLLossLayerTest() {
    delete blob_bottom_data1_;
    delete blob_bottom_data2_;
  }
  Blob<Dtype>* const blob_bottom_data1_;
  Blob<Dtype>* const blob_bottom_data2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(SoftmaxKLLossLayerTest, Dtypes);


TYPED_TEST(SoftmaxKLLossLayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  SoftmaxKLLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), -1, -1, -1);
}

TYPED_TEST(SoftmaxKLLossLayerTest, TestGradientGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  SoftmaxKLLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), -1, -1, -1);
}

}  // namespace caffe
