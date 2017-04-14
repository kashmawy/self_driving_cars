#include <iostream>
#include "tools.h"

#define ALPHA 0.0001

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  cout << "RMSE" << endl;

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  for (unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  cout << "Jacobian" << endl;

  MatrixXd Hj(3, 4);

  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  if (fabs(px) < ALPHA and fabs(py) < ALPHA){
	  px = ALPHA;
	  py = ALPHA;
  }

  float c1 = px*px + py*py;
  float c2 = sqrt(c1);
  float c3 = (c1 * c2);

  if (fabs(c1) < ALPHA) {
    c1 = ALPHA;
  }

  if (fabs(c2) < ALPHA) {
    c2 = ALPHA;
  }

  if (fabs(c3) < ALPHA) {
    c3 = ALPHA;
  }

  Hj << (px/c2), (py/c2), 0, 0,
        -(py/c1), (px/c1), 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}
