// Copyright 2014 Tobias Domhan

#include <stdlib.h>

#include <ctime>
#include <cmath>

#include <cstring>
#include <algorithm>
#include <iostream>
#include <stdio.h>

#include "gtest/gtest.h"
#include "caffe/common.hpp"

#include "caffe/proto/caffe.pb.h"

#include "caffe/test/test_caffe_main.hpp"

#include "caffe/net.hpp"
#include "caffe/solver.hpp"

namespace caffe {

  typedef double Dtype;

  TEST(TestTerminationCriterion, MaxIter) {
    MaxIterTerminationCriterion<Dtype> criterion(3);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyIteration(1);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyIteration(2);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyIteration(3);
    EXPECT_TRUE(criterion.IsCriterionMet());
  }

  TEST(TestTerminationCriterion, TestDivergenceDetectionNan) {
    DivergenceDetectionTerminationCriterion<Dtype> criterion;
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyValidationLoss(100);
    EXPECT_FALSE(criterion.IsCriterionMet());

    criterion.NotifyValidationLoss(0.1);
    EXPECT_FALSE(criterion.IsCriterionMet());

    criterion.NotifyValidationLoss(0.7);
    EXPECT_FALSE(criterion.IsCriterionMet());

    criterion.NotifyValidationLoss(-nan(""));
    EXPECT_TRUE(criterion.IsCriterionMet());

    criterion.NotifyValidationLoss(nan(""));
    EXPECT_TRUE(criterion.IsCriterionMet());
  }

  TEST(TestTerminationCriterion, TestDivergenceDetectionLoss) {
    DivergenceDetectionTerminationCriterion<Dtype> criterion;
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyValidationLoss(2.3);
    EXPECT_FALSE(criterion.IsCriterionMet());


    criterion.NotifyValidationLoss(1.3);
    EXPECT_FALSE(criterion.IsCriterionMet());

    criterion.NotifyValidationLoss(0.001);
    EXPECT_FALSE(criterion.IsCriterionMet());

    //let the loss explode:
    criterion.NotifyValidationLoss(80);
    EXPECT_TRUE(criterion.IsCriterionMet());
  }

  TEST(TestTerminationCriterion, TestDivergenceDetectionLoss2) {
    DivergenceDetectionTerminationCriterion<Dtype> criterion;
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyValidationLoss(80);
    EXPECT_FALSE(criterion.IsCriterionMet());


    criterion.NotifyValidationLoss(2);
    EXPECT_FALSE(criterion.IsCriterionMet());

    criterion.NotifyValidationLoss(0.001);
    EXPECT_FALSE(criterion.IsCriterionMet());

    //let the loss explode:
    criterion.NotifyValidationLoss(8000);
    EXPECT_TRUE(criterion.IsCriterionMet());
  }
  
  TEST(TestTerminationCriterion, TestAccuracy) {
    TestAccuracyTerminationCriterion<Dtype> criterion(3);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyValidationAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());

    //first countdown
    criterion.NotifyValidationAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //second countdown
    criterion.NotifyValidationAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //reset
    criterion.NotifyValidationAccuracy(0.6);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //first countdown
    criterion.NotifyValidationAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //second countdown
    criterion.NotifyValidationAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //third countdown
    criterion.NotifyValidationAccuracy(0.5);
    
    EXPECT_TRUE(criterion.IsCriterionMet());
  }

  TEST(TestTerminationCriterion, ExternalRunInBackgroundTerminationCriterion) {
    int run_every = 10;
    int ret;
    ExternalRunInBackgroundTerminationCriterion<Dtype> criterion("touch test", run_every);

    EXPECT_FALSE(criterion.IsCriterionMet());

    criterion.NotifyValidationAccuracy(0.5);
    EXPECT_TRUE(std::ifstream("learning_curve.txt"));

    criterion.NotifyIteration(run_every+1);
    EXPECT_TRUE(std::ifstream("termination_criterion_running"));
    criterion.NotifyIteration(run_every+2);
    EXPECT_FALSE(criterion.IsCriterionMet());

    ret = system("rm termination_criterion_running");

    criterion.NotifyIteration(run_every+3);
    EXPECT_FALSE(criterion.IsCriterionMet());

    criterion.NotifyIteration(2*run_every+1);

    ret = system("rm termination_criterion_running");
    ret = system("touch y_predict.txt");

    criterion.NotifyIteration(2*run_every+2);

    EXPECT_TRUE(criterion.IsCriterionMet());

    //make sure the touch was run:
    EXPECT_TRUE(std::ifstream("test"));
    ret = system("rm test");
  }


  TEST(TestTerminationCriterion, ExternalRunInBackgroundTerminationCriterionIsRunInBackground) {
    int run_every = 10;
    double epsilon_time = 1.;
    int ret;
    ExternalRunInBackgroundTerminationCriterion<Dtype> criterion("sleep 5", run_every);

    //check that the command is actually run in the background.

    clock_t begin = clock();
    criterion.NotifyIteration(run_every+1);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    EXPECT_TRUE(elapsed_secs < epsilon_time);
    LOG(INFO) << elapsed_secs;

    system("kill `pidof sleep`");
  }

  TEST(TestTerminationCriterion, ExternalRunInBackgroundTerminationCriterionIsRun) {
    int run_every = 10;
    int ret;
    ExternalRunInBackgroundTerminationCriterion<Dtype> criterion("touch test", run_every);

    //check that the command is actually run in the background.

    criterion.NotifyIteration(run_every+1);
    sleep(1);
    EXPECT_TRUE(std::ifstream("test"));
    ret = system("rm test");
  }


  TEST(TestTerminationCriterion, ExternalRunInBackgroundTerminationCriterionIsKilled) {
    int run_every = 10;
    int ret;

    //make sure
    ret = system("pidof sleep");
    EXPECT_NE(ret, 0) << " " <<"WARNING: make sure there's no sleep process running at the moment!";

    {
        ExternalRunInBackgroundTerminationCriterion<Dtype> criterion("sleep 10000", run_every);
        criterion.NotifyIteration(run_every+1);
        sleep(1);
        //this is a little hacky, because it assumes only a single sleep instance is running at the moment
        system("pidof sleep > termination_criterion_running_pid");
        ret = system("pidof sleep");
        EXPECT_EQ(ret, 0);
    }
    ret = system("pidof sleep");

    //getting the pid of sleep should be false now, because it should have been killed
    EXPECT_NE(ret, 0);
  }



}  // namespace caffe
