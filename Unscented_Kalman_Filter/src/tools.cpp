#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * Calculate the RMSE here.
   */
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  // ... your code here

  //accumulate squared residuals
  for(int i=0; i < estimations.size(); ++i){
    VectorXd d = estimations[i] - ground_truth[i];
    d = d.array() * d.array();
    rmse += d;
  }
  rmse /= estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}

double Tools::EnsureAngle(double angle) {
  while (angle < -M_PI)
    angle += 2 * M_PI;
  while (angle > M_PI)
    angle -= 2 * M_PI;
  return angle;
}
