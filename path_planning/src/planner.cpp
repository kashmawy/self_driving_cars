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
Planner::Planner() {
    this->state = STATE::START;
}

Planner::~Planner() {}

double get_lane_d(double D){
  double d;
  if (D < 4.0) {
    d = 2.0;
  }
  else if ((D >= 4.0) && (D < 8.0)) {
    d = 6.0;
  }
  else {
    d = 10.0;
  }
  return d;
}

LANE get_lane(double d){
  LANE lane;
  if (d < 4.0) {
    lane = LANE::LEFT;
  }
  else if ((d >= 4.0) && (d < 8.0)) {
    lane = LANE::MIDDLE;
  }
  else {
    lane = LANE::RIGHT;
  }
  return lane;
}

double get_lane_d(LANE lane){
  double d;
  if (lane == LANE::LEFT) {
    d = 2.0;
  }
  else if (lane == LANE::MIDDLE) {
    d = 6.0;
  }
  else {
    d = 10.0;
  }
  return d;
}

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

void Planner::create_trajectory(Map& map, Road& road, Car& car, vector<vector<double>>& trajectory) {
    int current_points = trajectory[0].size();
    this->new_points = false;

    if (current_points < POINTS) {
        this->new_points = true;

        if (this->state == STATE::START) {
            this->start_car(car);
        } else if (this->state == STATE::KEEP_LANE) {

            if (road.safe_lane(car, car.get_lane())) {
                this->stay_in_lane(car);
            } else {
                LANE target_lane = road.lane_change_available(car);
                if (target_lane == car.get_lane()) {
                    this->reduce_speed(car);
                } else {
                    this->change_lane(car, target_lane);
                }
            }
        } else {
            LANE new_lane = get_lane(car.prev_d()[0]);
            if(road.safe_lane(car, new_lane)){
                this->stay_in_lane(car);
            } else {
                this->reduce_speed(car);
            }
        }
    }

    if (this->new_points) {
        this->estimate_new_points(map, trajectory);
    }
}

void Planner::start_car(Car& car){
  cout << "ACTION: start_car" << endl;
  this->n = 4*POINTS; // 4 cycles to start
  double target_v = SPEED_LIMIT*0.5;
  double target_s = car.get_s() + n * AT * target_v;;

  this->start_s = {car.get_s(), car.get_v(), 0.0};
  this->end_s= {target_s, target_v, 0.0};

  this->start_d = {get_lane_d(car.get_lane()), 0.0, 0.0};
  this->end_d = {get_lane_d(car.get_lane()), 0.0, 0.0};

  this->apply_action(car, car.get_lane(), car.get_lane());
}

void Planner::stay_in_lane(Car& car){
  cout << "ACTION: stay_in_lane" << endl;
  this->n = CYCLES*POINTS;
  double target_v = min(car.prev_s()[1]*1.3, SPEED_LIMIT);
  double target_s = car.prev_s()[0] + n * AT * target_v;

  this->start_s = {car.prev_s()[0], car.prev_s()[1], car.prev_s()[2]};
  this->end_s = {target_s, target_v, 0.0};

  double target_d = get_lane_d(car.prev_d()[0]);

  this->start_d = {get_lane_d(car.prev_d()[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  this->apply_action(car, get_lane(car.prev_d()[0]), get_lane(car.prev_d()[0]));
}

void Planner::reduce_speed(Car& car){
  cout << "ACTION: reduce_speed" << endl;
  this->n = CYCLES*POINTS;
  this->new_points = true;
  double target_v = max(car.prev_s()[1]*0.8, SPEED_LIMIT/2);
  double target_s = car.prev_s()[0] + n * AT * target_v;

  this->start_s = {car.prev_s()[0], car.prev_s()[1], car.prev_s()[2]};
  this->end_s = {target_s, target_v, 0.0};

  double target_d = get_lane_d(car.prev_d()[0]);

  this->start_d = {get_lane_d(car.prev_d()[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  this->apply_action(car, get_lane(car.prev_d()[0]), get_lane(car.prev_d()[0]));
}

void Planner::change_lane(Car& car, LANE target_lane){
  cout << "ACTION: reduce_speed" << endl;
  this->n = CYCLES*POINTS;
  this->new_points = true;
  double target_v = car.prev_s()[1];
  double target_s = car.prev_s()[0] + n * AT * target_v;

  this->start_s = {car.prev_s()[0], car.prev_s()[1], car.prev_s()[2]};
  this->end_s = {target_s, target_v, 0.0};

  double target_d = get_lane_d(target_lane);

  this->start_d = {get_lane_d(car.prev_d()[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  this->apply_action(car, get_lane(car.prev_d()[0]), get_lane(target_d));
}

void Planner::apply_action(Car& car, LANE current_lane, LANE target_lane){
  car.set_previous_s(this->end_s);
  car.set_previous_d(this->end_d);
  this->set_state(current_lane, target_lane);
}

void Planner::set_state(LANE current_lane, LANE target_lane){
  if (current_lane == target_lane){
    this->state = STATE::KEEP_LANE;
  } else {
    // not equal
    if(current_lane == LANE::LEFT){
      this->state = STATE::MOVE_RIGHT;
    } else if(current_lane == LANE::RIGHT){
      this->state = STATE::MOVE_LEFT;
    } else {
      if(target_lane == LANE::LEFT){
        this->state = STATE::MOVE_LEFT;
      } else {
        this->state = STATE::MOVE_RIGHT;
      }
    }
  }
}