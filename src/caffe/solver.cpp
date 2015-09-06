// Copyright 2014 BVLC and contributors.

#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cmath> 

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param)
    : net_() {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file)
    : net_() {
  SolverParameter param;
  ReadProtoFromTextFile(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  LOG(INFO) << "Initializing solver from parameters: " << std::endl
            << param.DebugString(); 
  param_ = param;
  if (param_.solver_mode() == SolverParameter_SolverMode_GPU &&
      param_.has_device_id()) {
    Caffe::SetDevice(param_.device_id());
  }
  Caffe::set_mode(Caffe::Brew(param_.solver_mode()));
  if (param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  if (param_.has_train_net_param()) {
    CHECK(!param_.has_train_net()) << "Either train_net_param or train_net may "
                                   << "be specified, but not both.";
    LOG(INFO) << "Creating training net specified in SolverParameter.";
    net_.reset(new Net<Dtype>(param_.train_net_param()));
  } else {
    CHECK(param_.has_train_net())
        << "Neither train_net nor train_net_param were specified.";
    LOG(INFO) << "Creating training net from file: " << param_.train_net();
    net_.reset(new Net<Dtype>(param_.train_net()));
  }
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_test_nets) {
    CHECK_EQ(param_.test_iter_size(), num_test_nets)
        << "test_iter must be specified for each test network.";
    CHECK_GT(param_.test_interval(), 0);
  }
  test_nets_.resize(num_test_nets);
  for (int i = 0; i < num_test_net_params; ++i) {
      LOG(INFO) << "Creating testing net (#" << i
                << ") specified in SolverParameter.";
      test_nets_[i].reset(new Net<Dtype>(param_.test_net_param(i)));
  }
  for (int i = 0, test_net_id = num_test_net_params;
       i < num_test_net_files; ++i, ++test_net_id) {
      LOG(INFO) << "Creating testing net (#" << test_net_id
                << ") from file: " << param.test_net(i);
      test_nets_[test_net_id].reset(new Net<Dtype>(param_.test_net(i)));
  }
  CHECK_GT(this->param_.termination_criterion().size(), 0) << "at least one termination criterion needed.";
  termination_criterions_.resize(this->param_.termination_criterion().size());
  for(int i=0; i < this->param_.termination_criterion().size(); i++) {
    if (this->param_.termination_criterion().Get(i) == SolverParameter::MAX_ITER) {
      termination_criterions_[i].reset(new MaxIterTerminationCriterion<Dtype >(param_.max_iter()));
    } else if (this->param_.termination_criterion().Get(i) == SolverParameter::TEST_ACCURACY) {
      CHECK(num_test_nets) << "Test network needed for TEST_ACCURACY termination criterion.";
      bool valid_net = false;
      for (int test_net_id = 0; test_net_id < test_nets_.size(); ++test_net_id) {
        if (test_nets_[test_net_id]->name() == "valid") {
          valid_net = true;
        }
      }
      CHECK(valid_net) << "Network with the name 'valid' needed for TEST_ACCURACY termination criterion.";
      termination_criterions_[i].reset(new TestAccuracyTerminationCriterion<Dtype >(param_.test_accuracy_stop_countdown()));
    } else if (this->param_.termination_criterion().Get(i) == SolverParameter::DIVERGENCE_DETECTION) {
      CHECK(num_test_nets) << "Test network needed for DIVERGENCE_DETECTION termination criterion.";
      bool valid_net = false;
      for (int test_net_id = 0; test_net_id < test_nets_.size(); ++test_net_id) {
        if (test_nets_[test_net_id]->name() == "valid") {
          valid_net = true;
        }
      }
      CHECK(valid_net) << "Network with the name 'valid' needed for DIVERGENCE_DETECTION termination criterion.";
      termination_criterions_[i].reset(new DivergenceDetectionTerminationCriterion<Dtype >());
    } else if (this->param_.termination_criterion().Get(i) == SolverParameter::EXTERNAL) {
      CHECK(param_.has_external_term_criterion_cmd()) << "external_term_criterion_cmd needed";
      CHECK(param_.has_external_term_criterion_num_iter()) << "external_term_criterion_num_iter needed";
      termination_criterions_[i].reset(new ExternalTerminationCriterion<Dtype >(
        param_.external_term_criterion_cmd(),
        param_.external_term_criterion_num_iter()
        ));
    } else if (this->param_.termination_criterion().Get(i) == SolverParameter::EXTERNAL_IN_BACKGROUND) {
      CHECK(param_.has_external_term_criterion_cmd()) << "external_term_criterion_cmd needed";
      CHECK(param_.has_external_term_criterion_num_iter()) << "external_term_criterion_num_iter needed";
      termination_criterions_[i].reset(new ExternalRunInBackgroundTerminationCriterion<Dtype >(
        param_.external_term_criterion_cmd(),
        param_.external_term_criterion_num_iter()
        ));
    }
  }
  LOG(INFO) << "Solver scaffolding done.";
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  Caffe::set_phase(Caffe::TRAIN);
  LOG(INFO) << "Solving " << net_->name();
  PreSolve();

  iter_ = 0;
  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }
  
  if (this->param_.has_load_weights_from()) {
    NetParameter net_param;
    LOG(INFO) << "Loading weights from " << this->param_.load_weights_from();
    ReadProtoFromBinaryFile(this->param_.load_weights_from().c_str(), &net_param);
    net_->CopyTrainedLayersFrom(net_param);
    net_->initialize_weights();
  }

  // Run a test pass before doing any training to avoid waiting a potentially
  // very long time (param_.test_interval() training iterations) to report that
  // there's not enough memory to run the test net and crash, etc.; and to gauge
  // the effect of the first training iterations.
  if (param_.test_interval()) {
    TestAll();
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  vector<Blob<Dtype>*> bottom_vec;
  do {
    iter_++;
    for (int i=0; i < termination_criterions_.size(); i++) {
      termination_criterions_[i]->NotifyIteration(iter_);
    }

    Dtype loss = net_->ForwardBackward(bottom_vec);
    ComputeUpdateValue();
    net_->Update();

    if (param_.display() && iter_ % param_.display() == 0) {
      //LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
      for(std::vector<std::pair<int, float> >::iterator it = net_->losses().begin(); it != net_->losses().end(); ++it)
        LOG(INFO) << "Iteration " << iter_ << ", loss layer " << net_->layer_names()[(*it).first] << " = " << (*it).second;
      LOG(INFO) << "Iteration " << iter_ << ", total loss = " << loss;
    }
    if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
      TestAll();
    }
    // Check if we need to do snapshot
    if (param_.snapshot() && iter_ % param_.snapshot() == 0) {
      Snapshot();
    }
  } while (!TerminationCriterionsMet());
  if (param_.snapshot_on_exit()) {
    //iter_--;
    Snapshot();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
bool Solver<Dtype>::TerminationCriterionsMet() {
  for (int i=0; i < termination_criterions_.size(); i++) {
    if (termination_criterions_[i]->IsCriterionMet()) {
      return true;
    }
  }
  return false;
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  time_t timer;
  timer = time(NULL);
  LOG(INFO) << "Test timestamp " << timer;
  for (int test_net_id = 0; test_net_id < test_nets_.size(); ++test_net_id) {
    Test(test_net_id);
  }
}


template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  // We need to set phase to test before running.
  Caffe::set_phase(Caffe::TEST);
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<Blob<Dtype>*> bottom_vec;
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_nets_[test_net_id]->Forward(bottom_vec, &iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << test_nets_[test_net_id]->name() << " test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    LOG(INFO) << test_nets_[test_net_id]->name() << " test score #" << i << ": "
        << test_score[i] / param_.test_iter().Get(test_net_id);
  }
  if (test_nets_[test_net_id]->name() == "valid") {
    double valid_accuracy = test_score[0] / param_.test_iter().Get(test_net_id);
    double valid_loss = test_score[1] / param_.test_iter().Get(test_net_id);
    for (int i=0; i < termination_criterions_.size(); i++) {
      termination_criterions_[i]->NotifyValidationAccuracy(valid_accuracy);
      termination_criterions_[i]->NotifyValidationLoss(valid_loss);
    }
  }
  Caffe::set_phase(Caffe::TRAIN);
}


template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  NetParameter net_param;
  // For intermediate results, we will also dump the gradient values.
  net_->ToProto(&net_param, param_.snapshot_diff());
  string filename(param_.snapshot_prefix());
  const int kBufferSize = 20;
  char iter_str_buffer[kBufferSize];
  snprintf(iter_str_buffer, kBufferSize, "_iter_%d", iter_);
  filename += iter_str_buffer;
  LOG(INFO) << "Snapshotting to " << filename;
  WriteProtoToBinaryFile(net_param, filename.c_str());
  SolverState state;
  SnapshotSolverState(&state);
  state.set_iter(iter_);
  state.set_learned_net(filename);
  filename += ".solverstate";
  LOG(INFO) << "Snapshotting solver state to " << filename;
  WriteProtoToBinaryFile(state, filename.c_str());
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  SolverState state;
  NetParameter net_param;
  ReadProtoFromBinaryFile(state_file, &state);
  if (state.has_learned_net()) {
    ReadProtoFromBinaryFile(state.learned_net().c_str(), &net_param);
    net_->CopyTrainedLayersFrom(net_param);
  }
  iter_ = state.iter();
  RestoreSolverState(state);
}


// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
// where base_lr, gamma, step and power are defined in the solver parameter
// protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    CHECK_GT(this->param_.stepsize(), 0) << "step size necessary.";
    int current_step = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), current_step);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "inv_bergstra_bengio") {
    CHECK_GT(this->param_.stepsize(), 0) << "step size necessary.";
    rate = (this->iter_ > this->param_.stepsize()) ? this->param_.base_lr() * Dtype(this->param_.stepsize()) / this->iter_
      : this->param_.base_lr();
  } else if (lr_policy == "arbitrary_steps") {
    CHECK_GE(this->param_.step_lr_size(), 1) << "need step_lr and step_iter for the fixed_steps policy";
    CHECK_EQ(this->param_.step_lr_size(), this->param_.step_iter_size()+1) << "must have one more step_lr than step_iter";
    int nsteps = this->param_.step_iter_size();
    int end_iter = 0;
    int i;
    for (i=0; i < nsteps && this->iter_ >= end_iter; ++i)
      end_iter += this->param_.step_iter().Get(i);
    if (this->iter_ >= end_iter)
      rate = this->param_.step_lr().Get(i);
    else
      rate = this->param_.step_lr().Get(i-1);
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}


template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  history_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const Blob<Dtype>* net_param = net_params[i].get();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
        net_param->num(), net_param->channels(), net_param->height(),
        net_param->width())));
  }
}


template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue() {
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<shared_ptr<Layer<Dtype> > >& net_layers = this->net_->layers();
  vector<float>& net_params_lr = this->net_->params_lr();
  vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
  vector<float>& net_params_weight_constraint = this->net_->params_weight_constraint();
  // get the learning rate
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  Dtype momentum = this->param_.momentum();
  Dtype weight_decay = this->param_.weight_decay();
  bool constrain_weights = this->param_.weight_constraint();
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
      Dtype weight_constraint = net_params_weight_constraint[param_id];
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
          net_params[param_id]->cpu_diff(), momentum,
          history_[param_id]->mutable_cpu_data());
      if (local_decay) {
        // add weight decay
        caffe_axpy(net_params[param_id]->count(),
            local_decay * local_rate,
            net_params[param_id]->cpu_data(),
            history_[param_id]->mutable_cpu_data());
      }
    }
    // rescale if necessary
    if (constrain_weights) {
        for (int layer_id = 0; layer_id < net_layers.size(); ++layer_id) {
            net_layers[layer_id]->normalize_weights(net_params_weight_constraint[layer_id]);
        }
    }
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // copy
      caffe_copy(net_params[param_id]->count(),
          history_[param_id]->cpu_data(),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
      caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
          net_params[param_id]->gpu_diff(), momentum,
          history_[param_id]->mutable_gpu_data());
      if (local_decay) {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay * local_rate,
            net_params[param_id]->gpu_data(),
            history_[param_id]->mutable_gpu_data());
      }
    }
    // rescale if necessary
    if (constrain_weights) {
        for (int layer_id = 0; layer_id < net_layers.size(); ++layer_id) {
            net_layers[layer_id]->normalize_weights(net_params_weight_constraint[layer_id]);
        }
    }
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // copy
      caffe_gpu_copy(net_params[param_id]->count(),
          history_[param_id]->gpu_data(),
          net_params[param_id]->mutable_gpu_diff());
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(SolverState* state) {
  state->clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state->add_history();
    history_[i]->ToProto(history_blob);
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverState(const SolverState& state) {
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

template <typename Dtype>
void MaxIterTerminationCriterion<Dtype >::NotifyIteration(int iter) {
  this->criterion_met_ = iter >= max_iter_;
}

template <typename Dtype>
void DivergenceDetectionTerminationCriterion<Dtype >::NotifyValidationAccuracy(Dtype test_accuracy) {
  if(!std::isfinite(test_accuracy)){
    this->criterion_met_ = true;
  }
}

template <typename Dtype>
void DivergenceDetectionTerminationCriterion<Dtype >::NotifyValidationLoss(Dtype loss) {
  if(!std::isfinite(loss)){
    this->criterion_met_ = true;
  } else {
    if (!initial_loss_set_){
      initial_loss_ = loss;
      initial_loss_set_ = true;
    } else {
      //check whether the current loss is an order of magnitude bigger than the initial loss:
      if(loss / initial_loss_ > 10) {
        this->criterion_met_ = true;
      }
    }
  }
}
  
template <typename Dtype>
void TestAccuracyTerminationCriterion<Dtype >::NotifyValidationAccuracy(Dtype test_accuracy) {
  if (test_accuracy > best_accuracy_) {
    //reset countdown
    count_down_ = test_accuracy_stop_countdown_;
    this->criterion_met_ = false;
    best_accuracy_ = test_accuracy;
  } else {
    --count_down_;
    if (count_down_ <= 0) {
      this->criterion_met_ = true;
    } else {
      this->criterion_met_ = false;
    }
  }
}

template <typename Dtype>
ExternalTerminationCriterion<Dtype >::ExternalTerminationCriterion(const std::string& cmd,
    int run_every_x_iterations)
 : cmd_(cmd),
   run_every_x_iterations_(run_every_x_iterations),
   learning_curve_file_("learning_curve.txt") {
}

template <typename Dtype>
void ExternalTerminationCriterion<Dtype >::NotifyValidationAccuracy(Dtype test_accuracy) {
  learning_curve_file_ << test_accuracy << std::endl;
  learning_curve_file_.flush();
}

template <typename Dtype>
void ExternalTerminationCriterion<Dtype >::NotifyIteration(int iter) {
  if (iter % run_every_x_iterations_ == 0) {
    run();
  }
}

template <typename Dtype>
void ExternalTerminationCriterion<Dtype >::run() {
  int ret = system(cmd_.c_str());
  if (ret) {
    LOG(INFO) << "external termination criterion met.";
    this->criterion_met_ = true;
  } else {
    this->criterion_met_ = false;
  }
}


template <typename Dtype>
ExternalRunInBackgroundTerminationCriterion<Dtype >::ExternalRunInBackgroundTerminationCriterion(const std::string& cmd,
    int run_every_x_iterations)
 : cmd_(cmd),
   run_every_x_iterations_(run_every_x_iterations),
   learning_curve_file_("learning_curve.txt"),
   iter_of_next_run_(run_every_x_iterations) {
    //add & character so that it's executed in the background
    cmd_.append(" &");

    //make sure y_predict.txt doesn't exist, because we will base the termination
    //decision on the presence of that file later on.
    if (std::ifstream("y_predict.txt")) {
      remove("y_predict.txt");
    }
    if (std::ifstream("termination_criterion_running")) {
      remove("termination_criterion_running");
    }
}

template <typename Dtype>
ExternalRunInBackgroundTerminationCriterion<Dtype >::~ExternalRunInBackgroundTerminationCriterion() {
  if (std::ifstream("termination_criterion_running_pid")) {
    LOG(INFO) << "aborting termination_criterion";
    system("kill `cat termination_criterion_running_pid`");
    remove("termination_criterion_running_pid");
  }
  if (std::ifstream("termination_criterion_running")) {
    remove("termination_criterion_running");
  }
}

template <typename Dtype>
void ExternalRunInBackgroundTerminationCriterion<Dtype >::NotifyValidationAccuracy(Dtype test_accuracy) {
  learning_curve_file_ << test_accuracy << std::endl;
  learning_curve_file_.flush();
}

template <typename Dtype>
void ExternalRunInBackgroundTerminationCriterion<Dtype >::NotifyIteration(int iter) {
  //check if terminated, by checking the if the flag file exists.
  if (std::ifstream("termination_criterion_running")) {
      //wait for it to complete...
  } else {
    //by convention if y_predict.txt was created the termination criterion was reached.
    if (std::ifstream("y_predict.txt")) {
      this->criterion_met_ = true;
    } else {
      this->criterion_met_ = false;
      if (iter >= iter_of_next_run_) {
        run();
        iter_of_next_run_ += run_every_x_iterations_;
      }
    }
  }
}

template <typename Dtype>
void ExternalRunInBackgroundTerminationCriterion<Dtype >::run() {
  //create file that indicates the termination criterion is running right now
  std::ofstream outfile ("termination_criterion_running");
  outfile << "running" << std::endl;
  outfile.close();

  //kick off the termination criterion.
  int result = system(cmd_.c_str());
}


INSTANTIATE_CLASS(Solver);
INSTANTIATE_CLASS(SGDSolver);
INSTANTIATE_CLASS(MaxIterTerminationCriterion);
INSTANTIATE_CLASS(TestAccuracyTerminationCriterion);
INSTANTIATE_CLASS(ExternalTerminationCriterion);
INSTANTIATE_CLASS(ExternalRunInBackgroundTerminationCriterion);

}  // namespace caffe
