// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

template <typename Dtype>
__global__ void set_to(const int n, const Dtype alpha, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = alpha;
  }
}

namespace caffe {

template <typename Dtype>
Dtype LabelToOnehotLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
//   const Dtype* bottom_data = bottom[0]->gpu_data();
//   Dtype* top_data = (*top)[0]->mutable_gpu_data();
//   const int num = bottom[0]->num();
//   const int count = (*top)[0]->count();
//   const int num_output = count / num;
//   int label;
//   memset(top_data, 0, sizeof(Dtype) * count);
//   for (int i = 0; i < num; ++i) {
//     LOG(FATAL) << "Boom!";
//     label = static_cast<int>(bottom_data[i]);
//     LOG(FATAL) << "Label " << label;
//     if (label >=0 && label < num_output)
//       top_data[i + num * label] = 1.;
//     else
//       LOG(FATAL) << "Label " << label << " outside of expected range [0," << num_output-1 << "]";
//   }
//   return Dtype(0);
//   const Dtype* bottom_data = bottom[0]->gpu_data();
//   Dtype* top_data = (*top)[0]->mutable_gpu_data();
//   const int num = bottom[0]->num();
//   const int count = (*top)[0]->count();
//   const int dim = count / num;
//   int label;
//   set_to<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
//       count, Dtype(0), top_data);
//   for (int i = 0; i < num; ++i) {
//     label = int(bottom_data[i]);
//     if (label >=0 && label < dim)
// //       top_data[i + num * label] = 1.;
//       top_data[i * dim + label] = 1.;
//     else
//       LOG(FATAL) << "Label " << label << " outside of expected range [0," << dim-1 << "]";
//   }
//   return Dtype(0);

  return Forward_cpu(bottom, top);
}

template <typename Dtype>
void LabelToOnehotLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
}

INSTANTIATE_CLASS(LabelToOnehotLayer);


}  // namespace caffe
