#include "kalman_filter.h"
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in,
                        MatrixXd &P_in,
                        MatrixXd &F_in,
                        MatrixXd &H_in,
                        MatrixXd &Hj_in,
                        MatrixXd &R_laser_in,
                        MatrixXd &R_radar_in,
                        MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  Hj_ = Hj_in;
  R_laser_ = R_laser_in;
  R_radar_ = R_radar_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  UpdateCommon(H_, R_laser_, y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  Tools tools;
  Hj_ = tools.CalculateJacobian(x_);

  double rho = sqrt(x_(0) * x_(0) + x_(1) * x_(1));
  double theta = atan(x_(1) / x_(0));
  double rho_dot = (x_(0) * x_(2) + x_(1) * x_(3)) / rho;
  VectorXd h = VectorXd(3);
  h << rho, theta, rho_dot;
  VectorXd y = z - h;

  UpdateCommon(Hj_, R_radar_, y);
}

void KalmanFilter::UpdateCommon(const MatrixXd &H, const MatrixXd &R, const VectorXd &y) {
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * S.inverse();

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H) * P_;
}
