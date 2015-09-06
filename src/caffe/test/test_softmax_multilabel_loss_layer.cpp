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
class SoftmaxMultilabelLossLayerTest : public ::testing::Test {
 protected:
  SoftmaxMultilabelLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    int num = blob_bottom_label_->num();
    int channels = blob_bottom_label_->channels();
    int width = blob_bottom_label_->width();
    int height = blob_bottom_label_->height();
    for (int i = 0; i < num; ++i) {
      for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
          blob_bottom_label_->mutable_cpu_data()[((i*channels + caffe_rng_rand() % 5)*width + x)*height + y] += 0.5;
          blob_bottom_label_->mutable_cpu_data()[((i*channels + caffe_rng_rand() % 5)*width + x)*height + y] += 0.5;
        }
      }
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
  }
  virtual ~SoftmaxMultilabelLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(SoftmaxMultilabelLossLayerTest, Dtypes);

// Gradients tests may not work because for convenience we normalize the loss to have minimum value 0 (effectively turning it into KL-divergence), 
// which affects the numerically computed gradient. Comment out couple of lines in the implementation of the layer to avoid this.

TYPED_TEST(SoftmaxMultilabelLossLayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  SoftmaxMultilabelLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), -1, -1, -1);
}

TYPED_TEST(SoftmaxMultilabelLossLayerTest, TestGradientGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  SoftmaxMultilabelLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), -1, -1, -1);
}

}  // namespace caffe
