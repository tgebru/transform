// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void InnerProductOrthLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  // Figure out the dimensions
  M_ = bottom[0]->num();
  K_ = bottom[0]->count() / bottom[0]->num();
  N_ = num_output;
  (*top)[0]->Reshape(bottom[0]->num(), num_output, 1, 1);
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
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, N_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  // Setting up the bias multiplier
  if (bias_term_) {
    bias_multiplier_.reset(new SyncedMemory(M_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < M_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
  // Setting up variables needed for orthogonalization
//   LOG(INFO) << "Orth layer: N_=" << N_ << ", K_=" << K_;
  iter_ = 0;
  orth_step_ = this->layer_param_.orth_param().step();
  orth_method_ = this->layer_param_.orth_param().orth_method();
  max_num_iter_ = this->layer_param_.orth_param().max_num_iter();
  eps_ = this->layer_param_.orth_param().eps();
  col_norm_ = this->layer_param_.orth_param().col_norm();
  esmaeili_coeff_ = this->layer_param_.orth_param().esmaeili_coeff();
  transpose_ = this->layer_param_.orth_param().transpose();
  orth_before_iter_ = this->layer_param_.orth_param().orth_before_iter();
  if (N_ <= K_)
    min_error_ = Dtype(0);
  else
    min_error_ = std::sqrt(Dtype(N_ - K_));
  min_norm_ = this->layer_param_.orth_param().min_norm();
  max_norm_ = this->layer_param_.orth_param().max_norm();
  target_norm_ = this->layer_param_.orth_param().target_norm();
  this->gram_.Reshape(1, 1, N_, N_);
  this->kk_.Reshape(1, 1, N_, N_);
  this->id_.Reshape(1, 1, N_, N_);
  this->ak_.Reshape(1, 1, N_, K_);
  Dtype* id_data = this->id_.mutable_cpu_data();
  for (int i=0; i < N_; i++) {
    for (int j=0; j < N_; j++) {
      if (i==j)
        id_data[i * N_ + j] = 1.;
      else
        id_data[i * N_ + j] = 0.;
    }
  }
}

template <typename Dtype>
Dtype InnerProductOrthLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* weight = this->blobs_[0]->mutable_cpu_data();
  
  // Orthogonalizing
  if (Caffe::phase() == Caffe::TRAIN && orth_step_ > 0) {
    if (!(iter_ % orth_step_) && (orth_before_iter_ == 0 || iter_ < orth_before_iter_)) {
//       LOG(INFO) << "Orthogonalizing, iter=" << iter_;
      switch (orth_method_) {
        case OrthParameter_OrthMethod_ESMAEILI:
        {  
//           LOG(INFO) << "ESMAEILI";
          Dtype* gram = this->gram_.mutable_cpu_data();
          Dtype* kk = this->kk_.mutable_cpu_data();
          Dtype* ak = this->ak_.mutable_cpu_data();
          const Dtype* id = this->id_.cpu_data();
          Dtype error;
          caffe_scal(N_*K_, Dtype(1) / col_norm_, weight);
        
          for (int ni=0; ni<max_num_iter_; ni++) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, N_, K_, (Dtype)1.,
                weight, weight, (Dtype)0., gram);
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, N_, N_, esmaeili_coeff_,
                gram, gram, (Dtype)0., kk);
            caffe_axpy<Dtype>(N_*N_, -(1. + esmaeili_coeff_), gram, kk);
            caffe_axpy<Dtype>(N_*N_, Dtype(2), id, kk);            
            caffe_axpy<Dtype>(N_*N_, Dtype(-1), id, gram);
            error = caffe_cpu_norm2(N_*N_, gram);
//             LOG(INFO) << "Iter " << ni+1 << "  ||Gram - id||=" << error;
            if (error < min_error_ + eps_)
              ni = max_num_iter_;
            else {
              LOG(INFO) << "Iter " << iter_ << "." << ni <<"  ||Gram - id||=" << error;
              caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, K_, N_, (Dtype)1.,
                kk, weight, (Dtype)0., ak);
              caffe_copy(N_*K_, ak, weight);
            }  
          }
          
          caffe_scal(N_*K_, col_norm_, weight);
          
          break;
        }
        case OrthParameter_OrthMethod_NORM_L2:
        {
          normalize_weights(min_norm_, max_norm_, target_norm_);
          break;
        }
        case OrthParameter_OrthMethod_NORM_L1:
        {
          normalize_weights_l1(min_norm_, max_norm_, target_norm_);
          break;
        }
        case OrthParameter_OrthMethod_NORM_L1_NONNEG:
        {
          for (int i=0; i < this->blobs_[0]->count(); ++i)
            if (weight[i] < 0.)
              weight[i] = 0.;
          normalize_weights_l1(min_norm_, max_norm_, target_norm_);
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
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
  return Dtype(0);
}

template <typename Dtype>
void InnerProductOrthLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  // Gradient with respect to weight
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
      top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
  if (bias_term_) {
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), (Dtype)0.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down) {
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        (*bottom)[0]->mutable_cpu_diff());
  }
}

template <typename Dtype>
void InnerProductOrthLayer<Dtype>::normalize_weights(Dtype mnorm) {
  Dtype *weight = 0;
  int M = this->blobs_[0]->height();
  int N = this->blobs_[0]->width();
  int off = this->blobs_[0]->offset(0, 0, 0, 1);
  switch (Caffe::mode()) {
  case Caffe::CPU:
      // apply the constraint to each column
      weight = this->blobs_[0]->mutable_cpu_data();
      for (int i = 0; i < N; ++i) {
          // compute l2 norm
          Dtype nrm = caffe_cpu_norm2(M, weight, N);
          if (nrm > mnorm) {
              // and scale
              caffe_scal(M, mnorm / (nrm + Dtype(1e-7)), weight, N);
          }
          weight += off;
      }
      break;
  case Caffe::GPU:
      // apply the constraint to each column
      weight = this->blobs_[0]->mutable_gpu_data();
      for (int i = 0; i < N; ++i) {
          // compute l2 norm
          Dtype nrm = caffe_gpu_norm2(M, weight, N);
          if (nrm > mnorm) {
              // and scale
              caffe_gpu_scal(M, mnorm / (nrm + Dtype(1e-7)), weight, N);
          }
          weight += off;
      }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    break;
  }
}

template <typename Dtype>
void InnerProductOrthLayer<Dtype>::normalize_weights(Dtype min_norm, Dtype max_norm, Dtype target_norm) {
  Dtype *weight = 0;
  int M = this->blobs_[0]->height();
  int N = this->blobs_[0]->width();
  int off = this->blobs_[0]->offset(0, 0, 0, 1);
  switch (Caffe::mode()) {
  case Caffe::CPU:
      // apply the constraint to each column
      weight = this->blobs_[0]->mutable_cpu_data();
      for (int i = 0; i < N; ++i) {
          // compute l2 norm
          Dtype nrm = caffe_cpu_norm2(M, weight, N);
          if (nrm > max_norm || nrm < min_norm) {
              // and scale
              caffe_scal(M, target_norm / (nrm + Dtype(1e-7)), weight, N);
          }
          weight += off;
      }
      break;
  case Caffe::GPU:
      // apply the constraint to each column
      weight = this->blobs_[0]->mutable_gpu_data();
      for (int i = 0; i < N; ++i) {
          // compute l2 norm
          Dtype nrm = caffe_gpu_norm2(M, weight, N);
          if (nrm > max_norm || nrm < min_norm) {
              // and scale
              caffe_gpu_scal(M, target_norm / (nrm + Dtype(1e-7)), weight, N);
          }
          weight += off;
      }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    break;
  }
}

template <typename Dtype>
void InnerProductOrthLayer<Dtype>::normalize_weights_l1(Dtype min_norm, Dtype max_norm, Dtype target_norm) {
  Dtype *weight = 0;
  int M = this->blobs_[0]->height();
  int N = this->blobs_[0]->width();
  int off = this->blobs_[0]->offset(0, 0, 0, 1);
  switch (Caffe::mode()) {
  case Caffe::CPU:
      // apply the constraint to each column
      weight = this->blobs_[0]->mutable_cpu_data();
      for (int i = 0; i < N; ++i) {
          // compute l1 norm
          Dtype nrm = caffe_cpu_asum(M, weight, N);
          if (nrm > max_norm || nrm < min_norm) {
              // and scale
              caffe_scal(M, target_norm / (nrm + Dtype(1e-7)), weight, N);
          }
          weight += off;
      }
      break;
  case Caffe::GPU:
      // apply the constraint to each column
//       LOG(INFO) << "L1 norm on GPU";
      weight = this->blobs_[0]->mutable_gpu_data();
      for (int i = 0; i < N; ++i) {
          // compute l1 norm
          Dtype nrm;
          caffe_gpu_asum(M, weight, &nrm, N);
          if (nrm > max_norm || nrm < min_norm) {
              // and scale
              caffe_gpu_scal(M, target_norm / (nrm + Dtype(1e-7)), weight, N);
          }
          weight += off;
      }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    break;
  }
}

INSTANTIATE_CLASS(InnerProductOrthLayer);

}  // namespace caffe
