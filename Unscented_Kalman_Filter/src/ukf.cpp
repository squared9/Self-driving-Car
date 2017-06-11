#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

#define EPSILON 1E-4

//#define DO_DEBUG

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  //set state dimension
  n_x_ = 5;

  //set augmented dimension
  n_aug_ = 7;

  //define spreading parameter
  lambda_ = 3 - n_aug_;

  //create vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);

  //set weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  weights_(1) = 1 / (2 * (lambda_ + n_aug_));
  for (int i = 2; i < 2 * n_aug_ + 1; i++)
    weights_(i) = weights_(i - 1);

  //create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // NIS
  NIS_r_ = vector<double>();
  NIS_l_ = vector<double>();
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
#ifdef DO_DEBUG
  cout << "Process measurement start" << endl;
#endif

  if (!is_initialized_) {
    // initialization
    x_ << 1, 1, 1, 1, 1;
    P_ = MatrixXd::Identity(5, 5);
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      const double px = meas_package.raw_measurements_(0);
      const double py = meas_package.raw_measurements_(1);
      x_ << px, py, 0, 0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      const double rho = meas_package.raw_measurements_(0);
      const double psi = meas_package.raw_measurements_(1);
      const double rhod = meas_package.raw_measurements_(2);
      // polar to Cartesian
      const double px = rho * cos(psi);
      const double py = rho * sin(psi);
      const double vx = rhod * cos(psi);
      const double vy = rhod * sin(psi);
      const double psid = (fabs(px) < EPSILON && fabs(py) < EPSILON)? 0: atan2(py, px);
      x_ << px, py, vx, vy, 0;
    } else {
      // yet unknown sensor type
      cout << "Unknown sensor type " << meas_package.sensor_type_ << ". Ignoring";
    }
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  } else {
    // UKF sequence
    const double delta_t = ((double) (meas_package.timestamp_ - time_us_)) / 1E6;
    // first predict
    Prediction(delta_t);
    // then update from measurement
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      UpdateLidar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      UpdateRadar(meas_package);
    } else {
      // yet unknown sensor type
      cout << "Unknown sensor type " << meas_package.sensor_type_ << ". Ignoring";
    }
  }
  time_us_ = meas_package.timestamp_;
#ifdef DO_DEBUG
  cout << "Process measurement end" << endl;
#endif
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
#ifdef DO_DEBUG
  cout << "Prediction start" << endl;
#endif

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  //create augmented covariance matrix
  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();

  //create augmented Sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    VectorXd offset = sqrt(n_aug_ + lambda_) * A_aug.col(i);
    Xsig_aug.col(i + 1) = x_aug + offset;
    Xsig_aug.col(n_aug_ + i + 1) = x_aug - offset;
  }

  // Sigma point prediction
  const double dts = delta_t * delta_t / 2;
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x = Xsig_aug.col(i);
    const double px = x(0);
    const double py = x(1);
    const double nu = x(2);
    const double psi = x(3);
    const double psid = x(4);
    const double nu_a = x(5);
    const double nu_psidd = x(6);

    const double n_nu = nu + delta_t * nu_a;
    const double n_psi = psi + psid * delta_t + dts * nu_psidd;
    const double n_psid = psid + delta_t * nu_psidd;

    if (x(4) > EPSILON) {
      Xsig_pred_.col(i) << px + nu / psid * (sin(psi + psid * delta_t) - sin(psi)) + dts * cos(psi) * nu_a,
                           py + nu / psid * (-cos(psi + psid * delta_t) + cos(psi)) + dts * sin(psi) * nu_a,
                           n_nu,
                           n_psi,
                           n_psid;
    } else {
      Xsig_pred_.col(i) << px + nu * cos(psi) * delta_t + dts * cos(psi) * nu_a,
                           py + nu * sin(psi) * delta_t + dts * sin(psi) * nu_a,
                           n_nu,
                           n_psi,
                           n_psid;
    }
  }

  //predict state mean
  x_.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);

  //predict state covariance matrix
  P_.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd diff = Xsig_pred_.col(i) - x_;
    diff(3) = tools_.EnsureAngle(diff(3));
    P_ = P_ + weights_(i) * diff * diff.transpose();
  }
#ifdef DO_DEBUG
  cout << "Prediction end" << endl;
#endif
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
#ifdef DO_DEBUG
  cout << "Update LiDAR start" << endl;
#endif
  if (!use_laser_)
    return;

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 2;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x = Xsig_pred_.col(i);
    const double px = x(0);
    const double py = x(1);
    const double nu = x(2);
    const double psi = x(3);
    const double psid = x(4);
    const double rho = sqrt(px * px + py * py);
    Zsig.col(i) << px, py;
  }
  //calculate mean predicted measurement
  z_pred.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  //calculate measurement covariance matrix S
  MatrixXd R = MatrixXd(2, 2);
  R.fill(0);
  R(0, 0) = std_laspx_ * std_laspx_;
  R(1, 1) = std_laspy_ * std_laspy_;
  S.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * diff * diff.transpose();
  }
  S = S + R;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1);
  VectorXd zn = z - z_pred;

  x_ = x_ + K * zn;
  P_ = P_ - K * S * K.transpose();

  double eps = zn.transpose() * S.inverse() * zn;
  NIS_l_.push_back(eps);

#ifdef DO_DEBUG
  cout << "Update LiDAR end" << endl;
#endif
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
#ifdef DO_DEBUG
  cout << "Update radar start" << endl;
#endif
  if (!use_radar_)
    return;

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x = Xsig_pred_.col(i);
    const double px = x(0);
    const double py = x(1);
    const double nu = x(2);
    const double psi = x(3);
    const double psid = x(4);

    const double rho = sqrt(px * px + py * py);
    double phi = 0;
    if (py > EPSILON)
      phi = atan2(py, px);
    double rhod = 0;
    if (rho > EPSILON)
      rhod = (px * cos(psi) * nu + py * sin(psi) * nu) / rho;
    Zsig.col(i) << rho, phi, rhod;
  }
  //calculate mean predicted measurement
  z_pred.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  //calculate measurement covariance matrix S
  MatrixXd R = MatrixXd(3, 3);
  R.fill(0);
  R(0, 0) = std_radr_ * std_radr_;
  R(1, 1) = std_radphi_ * std_radphi_;
  R(2, 2) = std_radrd_ * std_radrd_;
  S.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd diff = Zsig.col(i) - z_pred;
    diff(1) = tools_.EnsureAngle(diff(1));
    S = S + weights_(i) * diff * diff.transpose();
  }
  S = S + R;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = tools_.EnsureAngle(x_diff(3));
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = tools_.EnsureAngle(z_diff(1));
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), meas_package.raw_measurements_(2);
  VectorXd zn = z - z_pred;
  zn(1) = tools_.EnsureAngle(zn(1));

  x_ = x_ + K * zn;
  P_ = P_ - K * S * K.transpose();

  double eps = zn.transpose() * S.inverse() * zn;
  NIS_r_.push_back(eps);

#ifdef DO_DEBUG
  cout << "Update radar end" << endl;
#endif
}
