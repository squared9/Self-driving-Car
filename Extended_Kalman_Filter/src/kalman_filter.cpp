#include "kalman_filter.h"

#define EPSILON 1E-3

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {
  I_ = MatrixXd::Identity(4, 4);
}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  VectorXd y = z - H_ * x_;
  MatrixXd HT = H_.transpose();
  MatrixXd S = H_ * P_ * HT + R_;
  MatrixXd K = P_ * HT * S.inverse();

  x_ = x_ + (K * y);
  P_ = (I_ - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  // convert Cartesian to polar coordinates
  VectorXd u = VectorXd(2);
  u << x_(0), x_(1);
//  double rho = u.norm();
  double rho = sqrt(x_(0) * x_(0) + x_(1) * x_(1));
  double phi = atan2(x_(1), x_(0));
  double rho_dot = 0;
  if (rho > EPSILON)
    rho_dot = (x_(0) * x_(2) + x_(1) * x_(3)) / rho;

  VectorXd z_next = VectorXd(3);
  z_next << rho, phi, rho_dot;

  VectorXd y = z - z_next;

  // normalize y_phi to [-pi, pi]
  while (y(1) < -M_PI)
    y(1) += 2 * M_PI;
  while (y(1) > M_PI)
    y(1) -= 2 * M_PI;

  MatrixXd HT = H_.transpose();
  MatrixXd S = H_ * P_ * HT + R_;
  MatrixXd K = P_ * HT * S.inverse();

  x_ = x_ + (K * y);
  P_ = (I_ - K * H_) * P_;
}
