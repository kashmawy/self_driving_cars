#include "planner.h"

using namespace std;

Planner::Planner() {
    this->state = STATE::START;
}

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

void Planner::create_new_trajectory_points(Map& map, vector<vector<double>>& trajectory) {
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
    this->new_path = false;

    if (current_points < POINTS) {
        this->new_path = true;

        if (this->state == STATE::START) {
            this->start_car(car);
        } else if (this->state == STATE::KEEP_LANE) {
            if (road.is_free_lane(car, car.get_lane())) {
                this->stay_in_lane(car);
            } else {
                LANE target_lane = road.get_free_lane(car);
                if (target_lane == car.get_lane()) {
                    this->decrease_speed(car);
                } else {
                    this->change_lane(car, target_lane);
                }
            }
        } else {
            LANE new_lane = get_lane(car.prev_d()[0]);
            if(road.is_free_lane(car, new_lane)){
                this->stay_in_lane(car);
            } else {
                this->decrease_speed(car);
            }
        }
    }

    if (this->new_path) {
        this->create_new_trajectory_points(map, trajectory);
    }
}

void Planner::start_car(Car& car){
  cout << "START CAR" << endl;

  this->n = 8 * POINTS;
  double target_v = SPEED_LIMIT * 0.7;
  double target_s = car.get_s() + n * AT * target_v;;

  this->start_s = {car.get_s(), car.get_v(), 0.0};
  this->end_s= {target_s, target_v, 0.0};

  this->start_d = {get_lane_d_from_lane(car.get_lane()), 0.0, 0.0};
  this->end_d = {get_lane_d_from_lane(car.get_lane()), 0.0, 0.0};

  this->apply_action(car, car.get_lane(), car.get_lane());
}

void Planner::stay_in_lane(Car& car){
  cout << "STAY IN LANE" << endl;

  this->n = CYCLES * POINTS;
  double target_v = min(car.prev_s()[1] * 1.3, SPEED_LIMIT);
  double target_s = car.prev_s()[0] + n * AT * target_v;

  this->start_s = {car.prev_s()[0], car.prev_s()[1], car.prev_s()[2]};
  this->end_s = {target_s, target_v, 0.0};

  double target_d = get_lane_d_from_d(car.prev_d()[0]);

  this->start_d = {get_lane_d_from_d(car.prev_d()[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  this->apply_action(car, get_lane(car.prev_d()[0]), get_lane(car.prev_d()[0]));
}

void Planner::decrease_speed(Car& car){
  cout << "DECREASE SPEED" << endl;

  this->n = CYCLES * POINTS;
  this->new_path = true;
  double target_v = max(car.prev_s()[1] * 0.9, SPEED_LIMIT/2.0);
  double target_s = car.prev_s()[0] + n * AT * target_v;

  this->start_s = {car.prev_s()[0], car.prev_s()[1], car.prev_s()[2]};
  this->end_s = {target_s, target_v, 0.0};

  double target_d = get_lane_d_from_d(car.prev_d()[0]);

  this->start_d = {get_lane_d_from_d(car.prev_d()[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  this->apply_action(car, get_lane(car.prev_d()[0]), get_lane(car.prev_d()[0]));
}

void Planner::change_lane(Car& car, LANE target_lane){
  cout << "CHANGE LANE" << endl;

  this->n = CYCLES * POINTS;
  this->new_path = true;
  double target_v = car.prev_s()[1];
  double target_s = car.prev_s()[0] + n * AT * target_v;

  this->start_s = {car.prev_s()[0], car.prev_s()[1], car.prev_s()[2]};
  this->end_s = {target_s, target_v, 0.0};

  double target_d = get_lane_d_from_lane(target_lane);

  this->start_d = {get_lane_d_from_d(car.prev_d()[0]), 0.0, 0.0};
  this->end_d = {target_d, 0.0, 0.0};

  this->apply_action(car, get_lane(car.prev_d()[0]), get_lane(target_d));
}

void Planner::apply_action(Car& car, LANE current_lane, LANE target_lane){
  car.set_previous_s(this->end_s);
  car.set_previous_d(this->end_d);
  this->update_state(current_lane, target_lane);
}

void Planner::update_state(LANE current_lane, LANE target_lane){
  if (current_lane == target_lane){
    this->state = STATE::KEEP_LANE;
  } else {
    if(current_lane == LANE::RIGHT){
      this->state = STATE::MOVE_LEFT;
    } else if(current_lane == LANE::LEFT){
      this->state = STATE::MOVE_RIGHT;
    } else {
      if(target_lane == LANE::LEFT){
        this->state = STATE::MOVE_LEFT;
      } else {
        this->state = STATE::MOVE_RIGHT;
      }
    }
  }
}