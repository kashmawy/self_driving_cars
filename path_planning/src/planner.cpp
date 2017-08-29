#include "planner.h"

using namespace std;

/*
    Jerk Minimizing Trajectory from initial state to final state in time T.

    start -> vehicles start location given
    end -> the desired end state for vehicle
    T -> The duration in seconds

    Output: An array of length 6, each corresponding to a coefficient in the polynomial
    s(t) = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5
*/
Planner::Planner() {}

Planner::~Planner() {}

vector<double> Planner::JMT(vector<double> start, vector<double> end, double T) {
    Eigen::MatrixXd A(3, 3);
    Eigen::MatrixXd B(3, 1);

    A << T*T*T, T*T*T*T, T*T*T*T*T,
         3*T*T, 4*T*T*T,5*T*T*T*T,
         6*T, 12*T*T, 20*T*T*T;

    B << end[0]-(start[0]+start[1]*T+.5*start[2]*T*T),
         end[1]-(start[1]+start[2]*T),
         end[2]-start[2];

    Eigen::MatrixXd Ai = A.inverse();
    Eigen::MatrixXd C = Ai * B;

    return {start[0], start[1], .5*start[2], C.data()[0], C.data()[1], C.data()[2]};
}

void Planner::estimate_new_points(Map& map, vector<vector<double>>& trajectory) {
    double T = this->n * AT;
    vector<double> poly_s = this->JMT(this->start_s, this->end_s, T);
    vector<double> poly_d = this->JMT(this->start_d, this->end_d, T);

    double t, next_s, next_d, mod_s, mod_d;
    vector<double> XY;

    for (int i = 0; i < n; ++i) {
        t = AT * i;
        next_s = 0.0;
        next_d = 0.0;

        for (int a = 0; a < poly_s.size(); ++a) {
            next_s += poly_s[a] * pow(t, a);
            next_d += poly_d[a] * pow(t, a);
        }

        mod_s = fmod(next_s, TRACK_DISTANCE);
        mod_d = fmod(next_d, ROAD_WIDTH);

        XY = map.getXY(mod_s, mod_d);

        trajectory[0].push_back(XY[0]);
        trajectory[1].push_back(XY[1]);
    }
}

