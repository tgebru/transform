// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <string>
#include <vector>
#include <fstream>

namespace caffe {

template <typename Dtype>
class TerminationCriterion;
  
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param);
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  virtual ~Solver() {}
  inline shared_ptr<Net<Dtype> > net() { return net_; }

 protected:
  // PreSolve is run before any solving iteration starts, allowing one to
  // put up some scaffold.
  virtual void PreSolve() {}
  // Get the update value for the current iteration.
  virtual void ComputeUpdateValue() = 0;
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  // if any of the criterions is met this is true
  bool TerminationCriterionsMet();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(SolverState* state) = 0;
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual void RestoreSolverState(const SolverState& state) = 0;

  SolverParameter param_;
  int iter_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;
  vector<shared_ptr<TerminationCriterion<Dtype > > > termination_criterions_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};


template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) {}
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) {}

 protected:
  virtual void PreSolve();
  Dtype GetLearningRate();
  virtual void ComputeUpdateValue();
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
  // history maintains the historical momentum data.
  vector<shared_ptr<Blob<Dtype> > > history_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
class TerminationCriterion {
public:
  TerminationCriterion() : criterion_met_(false) {};
  
  virtual bool IsCriterionMet() {return criterion_met_;};

  virtual void NotifyValidationAccuracy(Dtype test_accuracy) = 0;

  virtual void NotifyValidationLoss(Dtype loss) = 0;

  virtual void NotifyIteration(int iteration) = 0;
protected:
  bool criterion_met_;
};
  
template <typename Dtype>
class MaxIterTerminationCriterion : public TerminationCriterion<Dtype> {
public:
  MaxIterTerminationCriterion(int max_iter) : max_iter_(max_iter) {};

  virtual void NotifyValidationAccuracy(Dtype test_accuracy) {};

  virtual void NotifyValidationLoss(Dtype loss) {};
  
  virtual void NotifyIteration(int iteration);
private:
  int max_iter_;
};

template <typename Dtype>
class DivergenceDetectionTerminationCriterion : public TerminationCriterion<Dtype> {
  /**
    Checks whether the training has diverged.
  */
public:
  DivergenceDetectionTerminationCriterion() : initial_loss_set_(false){};
  
  virtual void NotifyValidationAccuracy(Dtype test_accuracy);

  virtual void NotifyValidationLoss(Dtype loss);
  
  virtual void NotifyIteration(int iteration) {};
  
private:
  Dtype initial_loss_;
  bool initial_loss_set_;
};
  
template <typename Dtype>
class TestAccuracyTerminationCriterion : public TerminationCriterion<Dtype> {
public:
  TestAccuracyTerminationCriterion(int test_accuracy_stop_countdown) :
    test_accuracy_stop_countdown_(test_accuracy_stop_countdown),
    count_down_(test_accuracy_stop_countdown),
    best_accuracy_(0.) {};
  
  virtual void NotifyValidationAccuracy(Dtype test_accuracy);

  virtual void NotifyValidationLoss(Dtype loss) {};
  
  virtual void NotifyIteration(int iteration) {};
  
private:
  const int test_accuracy_stop_countdown_;
  Dtype best_accuracy_;
  int count_down_;
};

template <typename Dtype>
class ExternalTerminationCriterion : public TerminationCriterion<Dtype> {
public:
  ExternalTerminationCriterion(const std::string& cmd, int run_every_x_iterations);
  
  virtual void NotifyValidationAccuracy(Dtype test_accuracy);

  virtual void NotifyValidationLoss(Dtype loss) {};
  
  virtual void NotifyIteration(int iteration);
  
private:

  void run();

  //command to call to check the termination criterion.
  std::string cmd_;
  std::ofstream learning_curve_file_;
  int run_every_x_iterations_;
};


template <typename Dtype>
class ExternalRunInBackgroundTerminationCriterion : public TerminationCriterion<Dtype> {
  /*
    Termination criterion is run in background parallelly to the caffe process.
    The status will be checked through files.

    The usage protocol is the following:

      1. Before the process is started a file called termination_criterion_running is created
      2. The termination criterion is responsible for deleting termination_criterion_running
         If this file is not deleted the termination criterion won't ever be run in the future!
      3. When the termination criterion is completed it will delete the file termination_criterion_running
         Furthermore the file y_predict.txt will be created in case the termination criterion is met.
  */
public:
  ExternalRunInBackgroundTerminationCriterion(const std::string& cmd, int run_every_x_iterations);
  
  ~ExternalRunInBackgroundTerminationCriterion();

  virtual void NotifyValidationAccuracy(Dtype test_accuracy);

  virtual void NotifyValidationLoss(Dtype loss) {};
  
  virtual void NotifyIteration(int iteration);
  
private:

  void run();

  //command to call to check the termination criterion.
  std::string cmd_;
  std::ofstream learning_curve_file_;
  int run_every_x_iterations_;
  int iter_of_next_run_;
};
  

}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
