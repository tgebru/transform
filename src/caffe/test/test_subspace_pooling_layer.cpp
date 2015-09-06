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
class SubspacePoolingLayerTest : public ::testing::Test {
 protected:
  SubspacePoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SubspacePoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForwardMax() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_size(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x inputs with channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    // and:
    //     [2 1 6 1 5]
    //     [1 2 2 2 2]
    //     [2 1 6 1 5]
    for (int i = 0; i <  num; i++) {
      int idx = blob_bottom_->offset(i);
      blob_bottom_->mutable_cpu_data()[idx +  0] = 1;
      blob_bottom_->mutable_cpu_data()[idx +  1] = 2;
      blob_bottom_->mutable_cpu_data()[idx +  2] = 5;
      blob_bottom_->mutable_cpu_data()[idx +  3] = 2;
      blob_bottom_->mutable_cpu_data()[idx +  4] = 3;
      blob_bottom_->mutable_cpu_data()[idx +  5] = 9;
      blob_bottom_->mutable_cpu_data()[idx +  6] = 4;
      blob_bottom_->mutable_cpu_data()[idx +  7] = 1;
      blob_bottom_->mutable_cpu_data()[idx +  8] = 4;
      blob_bottom_->mutable_cpu_data()[idx +  9] = 8;
      blob_bottom_->mutable_cpu_data()[idx + 10] = 1;
      blob_bottom_->mutable_cpu_data()[idx + 11] = 2;
      blob_bottom_->mutable_cpu_data()[idx + 12] = 5;
      blob_bottom_->mutable_cpu_data()[idx + 13] = 2;
      blob_bottom_->mutable_cpu_data()[idx + 14] = 3;

      idx += 15;
      blob_bottom_->mutable_cpu_data()[idx +  0] = 2;
      blob_bottom_->mutable_cpu_data()[idx +  1] = 1;
      blob_bottom_->mutable_cpu_data()[idx +  2] = 6;
      blob_bottom_->mutable_cpu_data()[idx +  3] = 1;
      blob_bottom_->mutable_cpu_data()[idx +  4] = 5;
      blob_bottom_->mutable_cpu_data()[idx +  5] = 1;
      blob_bottom_->mutable_cpu_data()[idx +  6] = 2;
      blob_bottom_->mutable_cpu_data()[idx +  7] = 2;
      blob_bottom_->mutable_cpu_data()[idx +  8] = 2;
      blob_bottom_->mutable_cpu_data()[idx +  9] = 2;
      blob_bottom_->mutable_cpu_data()[idx + 10] = 2;
      blob_bottom_->mutable_cpu_data()[idx + 11] = 1;
      blob_bottom_->mutable_cpu_data()[idx + 12] = 6;
      blob_bottom_->mutable_cpu_data()[idx + 13] = 1;
      blob_bottom_->mutable_cpu_data()[idx + 14] = 5;
    }
    SubspacePoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels / 2);
    EXPECT_EQ(blob_top_->height(), 3);
    EXPECT_EQ(blob_top_->width(), 5);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels / 2);
      EXPECT_EQ(blob_top_mask_->height(), 3);
      EXPECT_EQ(blob_top_mask_->width(), 5);
    }
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 1 channels of:
    //     [2 2 6 2 5]
    //     [9 4 2 4 8]
    //     [2 2 6 2 5]
    for (int i = 0; i < 15 * num; i += 15) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 6);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 6);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 5);
    }
    if (blob_top_vec_.size() > 1) {
      // TODOExpected mask output: 2x 1 channels of:
      //    TODO
      // DEBUG 
      // for (int i = 0; i < blob_top_mask_->count(); ++i) 
      //   std::cout << blob_top_mask_->cpu_data()[i] << " " << std::endl;
    }
  }

  void TestForwardAve() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_size(2);
    pooling_param->set_stride(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
    blob_bottom_->Reshape(1, 2, 3, 3);
    // fill channels with c +:
    // [1 2 3]
    // [1 2 3]
    // [1 2 3]
    for (int c= 0; c <  2; c++) {
      int idx = blob_bottom_->offset(0, c);
      blob_bottom_->mutable_cpu_data()[idx +  0] = 1 + c;
      blob_bottom_->mutable_cpu_data()[idx +  1] = 2 + c;
      blob_bottom_->mutable_cpu_data()[idx +  2] = 3 + c;
      blob_bottom_->mutable_cpu_data()[idx +  3] = 1 + c;
      blob_bottom_->mutable_cpu_data()[idx +  4] = 2 + c;
      blob_bottom_->mutable_cpu_data()[idx +  5] = 3 + c;
      blob_bottom_->mutable_cpu_data()[idx +  6] = 1 + c;
      blob_bottom_->mutable_cpu_data()[idx +  7] = 2 + c;
      blob_bottom_->mutable_cpu_data()[idx +  8] = 3 + c;
    }
    SubspacePoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &(blob_top_vec_));
    EXPECT_EQ(blob_top_->num(), 1);
    EXPECT_EQ(blob_top_->channels(), 1);
    EXPECT_EQ(blob_top_->height(), 3);
    EXPECT_EQ(blob_top_->width(), 3);
    layer.Forward(blob_bottom_vec_, &(blob_top_vec_));
    Dtype epsilon = 1e-5;
    EXPECT_NEAR(blob_top_->cpu_data()[0], 1.5, epsilon);
    EXPECT_NEAR(blob_top_->cpu_data()[1], 2.5, epsilon);
    EXPECT_NEAR(blob_top_->cpu_data()[2], 3.5, epsilon);
    EXPECT_NEAR(blob_top_->cpu_data()[3], 1.5, epsilon);
    EXPECT_NEAR(blob_top_->cpu_data()[4], 2.5, epsilon);
    EXPECT_NEAR(blob_top_->cpu_data()[5], 3.5, epsilon);
    EXPECT_NEAR(blob_top_->cpu_data()[6], 1.5, epsilon);
    EXPECT_NEAR(blob_top_->cpu_data()[7], 2.5, epsilon);
    EXPECT_NEAR(blob_top_->cpu_data()[8], 3.5, epsilon);
  }

};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(SubspacePoolingLayerTest, Dtypes);

TYPED_TEST(SubspacePoolingLayerTest, TestSetup) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(3);
  SubspacePoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels() / 2);
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(SubspacePoolingLayerTest, TestCPUForwardMax) {
  Caffe::set_mode(Caffe::CPU);
  this->TestForwardMax();
}

TYPED_TEST(SubspacePoolingLayerTest, TestGPUForwardMax) {
  Caffe::set_mode(Caffe::GPU);
  this->TestForwardMax();
}

TYPED_TEST(SubspacePoolingLayerTest, TestCPUForwardMaxTopMask) {
  Caffe::set_mode(Caffe::CPU);
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForwardMax();
}

TYPED_TEST(SubspacePoolingLayerTest, TestGPUForwardMaxTopMask) {
  Caffe::set_mode(Caffe::GPU);
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForwardMax();
}

TYPED_TEST(SubspacePoolingLayerTest, TestCPUGradientMax) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(3);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::CPU);
  SubspacePoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(SubspacePoolingLayerTest, TestGPUGradientMax) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(3);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::GPU);
  SubspacePoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}



TYPED_TEST(SubspacePoolingLayerTest, TestCPUForwardAve) {
  Caffe::set_mode(Caffe::CPU);
  this->TestForwardAve();
}

TYPED_TEST(SubspacePoolingLayerTest, TestGPUForwardAve) {
  Caffe::set_mode(Caffe::GPU);
  this->TestForwardAve();
}

TYPED_TEST(SubspacePoolingLayerTest, TestCPUGradientAve) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(3);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  Caffe::set_mode(Caffe::CPU);
  SubspacePoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}


TYPED_TEST(SubspacePoolingLayerTest, TestGPUGradientAve) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(3);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  Caffe::set_mode(Caffe::GPU);
  SubspacePoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}


}  // namespace caffe
