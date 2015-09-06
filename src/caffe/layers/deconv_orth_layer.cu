// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype DeConvolutionOrthLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* weight = this->blobs_[0]->mutable_gpu_data();
  //const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();

  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();

  
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int bottom_offset = M_ * N_;
  
  // Orthogonalization  
  if (Caffe::phase() == Caffe::TRAIN && orth_step_ > 0) {
    if (!(iter_ % orth_step_) && (orth_before_iter_ == 0 || iter_ < orth_before_iter_)) {
//       LOG(INFO) << "Orthogonalizing, iter=" << iter_;
      switch (orth_method_) {
        case OrthParameter_OrthMethod_ESMAEILI:
        { 
          Dtype* gram = this->gram_.mutable_gpu_data();
          Dtype* kk = this->kk_.mutable_gpu_data();
          Dtype* ak = this->ak_.mutable_gpu_data();
          const Dtype* id = this->id_.gpu_data();
          Dtype error;

          caffe_gpu_scal(weight_offset * group_, Dtype(1) / col_norm_, weight);
          
          for (int g = 0; g < group_; ++g) {
//           LOG(INFO) << "ESMAEILI";                 
            for (int ni=0; ni<max_num_iter_; ni++) {
              caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, M_, K_, (Dtype)1.,
                  weight, weight, (Dtype)0., gram);
              caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, M_, M_, esmaeili_coeff_,
                  gram, gram, (Dtype)0., kk);
              caffe_gpu_axpy<Dtype>(M_*M_, -(1. + esmaeili_coeff_), gram, kk);
              caffe_gpu_axpy<Dtype>(M_*M_, 2., id, kk);            
              caffe_gpu_axpy<Dtype>(M_*M_, -1., id, gram);
              error = caffe_gpu_norm2(M_*M_, gram);
  //             LOG(INFO) << "Iter " << ni+1 << "  ||Gram - id||=" << error;
//               if (error > 1e5)
                
              if (error < eps_)
                ni = max_num_iter_;
              else {
                LOG(INFO) << "Iter " << iter_ << "." << ni <<"  ||Gram - id||=" << error;
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, M_, (Dtype)1.,
                  kk, weight, (Dtype)0., ak);
                caffe_gpu_copy(M_*K_, ak, weight);
              }              
            }
            weight += weight_offset;
          }
          weight = this->blobs_[0]->mutable_gpu_data();
          caffe_gpu_scal(weight_offset * group_, col_norm_, weight);
          
          break;
        }
        case OrthParameter_OrthMethod_NORM_L2:
        {
          normalize_weights(min_norm_, max_norm_, target_norm_);
          break;
        }
        case OrthParameter_OrthMethod_NONE:
//           LOG(INFO) << "NONE";
          break;
        default:
          LOG(FATAL) << "Unknown orthogonalization method";
          break;  
      }
    }      
    iter_++;
  }
  
  // actually performing forward pass
  
  for (int n = 0; n < num_; ++n) {
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + weight_offset * g,
          bottom_data + bottom[0]->offset(n) + bottom_offset * g,
          (Dtype)0., col_data + col_offset * g);
      }
      // col2im forward to the top_data
      col2im_gpu(col_data, channels_, height_out_, width_out_, kernel_size_, pad_,
                 stride_, top_data + (*top)[0]->offset(n));
      // JTS: new way of adding the bias
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_, N_, 1,
                            (Dtype)1., this->blobs_[1]->gpu_data(),
                            reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
                            (Dtype)1., top_data + (*top)[0]->offset(n));
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
void DeConvolutionOrthLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
    //JTS fixed gradient wrt. bias, not sure about the group stuff ...
    for (int n = 0; n < num_; ++n) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
                              (Dtype)1., top_diff + top[0]->offset(n),
                              reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()), (Dtype)1.,
                              bias_diff);
    }
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


INSTANTIATE_CLASS(DeConvolutionOrthLayer);

}  // namespace caffe
