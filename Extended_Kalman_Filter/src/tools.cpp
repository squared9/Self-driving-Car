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

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
  */
  MatrixXd Hj(3,4);
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //check division by zero
  float i = px * px + py * py;
  float d = sqrt(i);
  if (fabs(d) < 1E-8)
    return Hj;
  float d3 = d * i;

  //compute the Jacobian matrix

  Hj << px/d, py/d, 0, 0,
          -py/i, px/i, 0, 0,
          py * (vx*py - vy*px)/d3, px*(vy*px - vx*py)/d3, px/d, py/d;

  return Hj;
}
