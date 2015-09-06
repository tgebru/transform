// Copyright 2014 BVLC and contributors.
/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <stdint.h>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
HDF5DataCouplesLayer<Dtype>::~HDF5DataCouplesLayer<Dtype>() {
    delete data_blob_;
    delete label_blob_;
}

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5DataCouplesLayer<Dtype>::LoadHDF5FileData(const char* filename,
  Blob<Dtype>* data_blob, Blob<Dtype>* label_blob) {

  LOG(INFO) << "Loading HDF5 file " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(ERROR) << "Failed opening HDF5 file" << filename;
    return;
  }

  const int MIN_DATA_DIM = 2;
  const int MAX_DATA_DIM = 4;
  hdf5_load_nd_dataset(
    file_id, "data",  MIN_DATA_DIM, MAX_DATA_DIM, data_blob);

  LOG(INFO) << "Loaded data blob " << data_blob->num() << "x" << data_blob->channels() << "x"
      << data_blob->height() << "x" << data_blob->width();
  
  const int MIN_LABEL_DIM = 1;
  const int MAX_LABEL_DIM = 2;
  hdf5_load_nd_dataset(
    file_id, "label", MIN_LABEL_DIM, MAX_LABEL_DIM, label_blob);
  
  LOG(INFO) << "Loaded label blob " << label_blob->num() << "x" << label_blob->channels() << "x"
      << label_blob->height() << "x" << label_blob->width();

  herr_t status = H5Fclose(file_id);
  CHECK_EQ(data_blob->num(), label_blob->num());
  LOG(INFO) << "Successully loaded " << data_blob->num() << " rows";
}

template <typename Dtype>
void HDF5DataCouplesLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "HDF5DataCouplesLayer takes no input blobs.";
  CHECK_EQ(top->size(), 3) << "HDF5DataCouplesLayer takes three blobs as output.";

  data_in_memory_ = this->layer_param_.hdf5_data_param().data_in_memory();
  CHECK(data_in_memory_) << "Currently only works with data_in_memory=true";
  
  match_ratio_ = static_cast<Dtype>(this->layer_param_.couples_param().match_ratio());

  // Read the source to parse the filenames.
  const string& source = this->layer_param_.hdf5_data_param().source();
  LOG(INFO) << "Loading data from " << source;

  data_blob_ = new Blob<Dtype>();
  label_blob_ = new Blob<Dtype>();
  LoadHDF5FileData(source.c_str(), data_blob_, label_blob_);
  
  in_data_num_ = data_blob_->num();
  scale_ = this->layer_param_.couples_param().scale();
 
  // Reshape blobs.
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  (*top)[0]->Reshape(batch_size, data_blob_->channels(),
                     data_blob_->width(), data_blob_->height());
  (*top)[1]->Reshape(batch_size, data_blob_->channels(),
                     data_blob_->width(), data_blob_->height());
  (*top)[2]->Reshape(batch_size, 1, 1, 1);
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  LOG(INFO) << "output labels size: " << (*top)[2]->num() << ","
      << (*top)[2]->channels() << "," << (*top)[2]->height() << ","
      << (*top)[2]->width();    
}

template <typename Dtype>
Dtype HDF5DataCouplesLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int num = (*top)[0]->num();
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  const int data_count = (*top)[0]->count();
  const int data_dim = data_count / num;
  const int label_dim = (*top)[2]->count() / num;

  int n1, n2;
  
  Dtype* top_data1 = (*top)[0]->mutable_cpu_data();
  Dtype* top_data2 = (*top)[1]->mutable_cpu_data();
  Dtype* top_label = (*top)[2]->mutable_cpu_data();
  const Dtype* in_data = data_blob_->cpu_data();
  const Dtype* in_label = label_blob_->cpu_data();
  
  caffe_rng_bernoulli(batch_size, match_ratio_, top_label);
  
  // Currently we simply suppose that each image is a separate class. 
  // TODO rewrite in a way that input images with equal labels are considered equivalent
  
  for (int i = 0; i < batch_size; ++i) {
    n1 = caffe_rng_rand() % in_data_num_;
    if (static_cast<int>(top_label[i]))
      n2 = n1;
    else {
      do {
        n2 = caffe_rng_rand() % in_data_num_;
      } while (n2 == n1);
    }
    memcpy(&top_data1[i * data_dim], &in_data[n1 * data_dim], sizeof(Dtype) * data_dim);
    memcpy(&top_data2[i * data_dim], &in_data[n2 * data_dim], sizeof(Dtype) * data_dim);
//     LOG(INFO) << "i=" << i << ", n1=" << n1 << ", n2=" << n2 << ", label=" << top_label[i];
  }
  
  if (std::fabs(scale_ - 1.) > 1e-3) {
    caffe_scal(data_count, scale_, top_data1);
    caffe_scal(data_count, scale_, top_data2);
  }
  
  return Dtype(0.);
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
void HDF5DataCouplesLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { }

INSTANTIATE_CLASS(HDF5DataCouplesLayer);

}  // namespace caffe
