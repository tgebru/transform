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
class UnPoolingLayerTest : public ::testing::Test {
 protected:
    UnPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_deconv_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    blob_bottom_deconv_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_unpool_vec_.push_back(blob_bottom_deconv_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~UnPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_deconv_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_deconv_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_unpool_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForward() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_size(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      blob_bottom_->mutable_cpu_data()[i +  5] = 9;
      blob_bottom_->mutable_cpu_data()[i +  6] = 4;
      blob_bottom_->mutable_cpu_data()[i +  7] = 1;
      blob_bottom_->mutable_cpu_data()[i +  8] = 4;
      blob_bottom_->mutable_cpu_data()[i +  9] = 8;
      blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 3;
    }
    PoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 2);
    EXPECT_EQ(blob_top_->width(), 4);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 2);
      EXPECT_EQ(blob_top_mask_->width(), 4);
    }
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [9 5 5 8]
    //     [9 5 5 8]
    for (int i = 0; i < 8 * num * channels; i += 8) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 8);
    }
    if (blob_top_vec_.size() > 1) {
      // Expected mask output: 2x 2 channels of:
      //     [5  2  2 9]
      //     [5 12 12 9]
      for (int i = 0; i < 8 * num * channels; i += 8) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  5);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  9);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4],  5);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7],  9);
      }
    }
    // unpooling should be the same as backward pass
    for (int i = 0; i < blob_top_->count(); ++i) {
        blob_top_->mutable_cpu_diff()[i] = 1.;
        // we set this to 1 so that it is equal to the gradient
        blob_top_->mutable_cpu_data()[i] = 1.;
    }
    layer.Backward(blob_top_vec_, true, &(blob_bottom_vec_));
    // test the unpooling
    UnPoolingLayer<Dtype> ulayer(layer_param);
    ulayer.SetUp(blob_top_vec_, &blob_bottom_unpool_vec_);
    ulayer.Forward(blob_top_vec_, &blob_bottom_unpool_vec_);
    for (int i = 0; i < blob_top_->count(); ++i) {
        EXPECT_EQ(blob_bottom_deconv_->cpu_data()[i], blob_bottom_->cpu_diff()[i]);
    }

    // next do a backward pass for unpooling 
    // which should recover the max pooling solution again
    for (int i = 0; i < blob_bottom_->count(); ++i) {
        blob_bottom_deconv_->mutable_cpu_diff()[i] = blob_bottom_->cpu_data()[i];
    }    
    ulayer.Backward(blob_bottom_unpool_vec_, true, &(blob_top_vec_));
    // recompute normal forward pass for comparison
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    for (int i = 0; i < blob_top_->count(); ++i) {
        EXPECT_EQ(blob_top_->cpu_diff()[i], blob_top_->cpu_data()[i]);
    }
    
    
    /* JTS some debug printing
    for (int i = 0; i < 8 * num * channels; i += 8) {
        cout <<  blob_top_->cpu_diff()[i+0] << " ";
        cout <<  blob_top_->cpu_diff()[i+1] << " ";
        cout <<  blob_top_->cpu_diff()[i+2] << " ";
        cout <<  blob_top_->cpu_diff()[i+3] << " ";
        cout <<  blob_top_->cpu_diff()[i+4] << " ";
        cout <<  blob_top_->cpu_diff()[i+5] << " ";
        cout <<  blob_top_->cpu_diff()[i+6] << " ";
        cout <<  blob_top_->cpu_diff()[i+7] << endl;
    }

    for (int i = 0; i < 15 * num * channels; i += 15) {
        cout <<  blob_bottom_deconv_->cpu_data()[i+0] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+1] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+2] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+3] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+4] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+5] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+6] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+7] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+8] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+9] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+10] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+11] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+12] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+13] << " ";
        cout <<  blob_bottom_deconv_->cpu_data()[i+14] << endl;
    }
    for (int i = 0; i < 15 * num * channels; i += 15) {
        cout <<  blob_bottom_->cpu_diff()[i+0] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+1] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+2] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+3] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+4] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+5] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+6] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+7] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+8] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+9] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+10] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+11] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+12] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+13] << " ";
        cout <<  blob_bottom_->cpu_diff()[i+14] << endl;
    }
    */
  }
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(UnPoolingLayerTest, Dtypes);

TYPED_TEST(UnPoolingLayerTest, TestCPUForwardMax) {
  Caffe::set_mode(Caffe::CPU);
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForward();
}

}  // namespace caffe
