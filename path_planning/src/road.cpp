#include "road.h"

using namespace std;

Road::Road() {}

Road::~Road() {}

void Road::update_road(vector<Car> left_lane, vector<Car> middle_lane, vector<Car> right_lane) {
    this->left_lane = left_lane;
    this->middle_lane = middle_lane;
    this->right_lane = right_lane;
}

vector<Car> Road::get_lane_cars(LANE lane) {
    if (lane == LANE::LEFT) {
        return this->left_lane;
    } else if (lane == LANE::MIDDLE) {
        return this->middle_lane;
    } else {
        return this->right_lane;
    }
}

bool Road::safe_lane(Car& car, LANE lane){
  vector<Car> r_car_lane = this->get_lane_cars(lane);
  bool safe = true;
  for (int i = 0; i < r_car_lane.size(); i++) {
    double distance = r_car_lane[i].get_s() - car.get_s();
    if(distance > 0 && distance < SAFETY_DISTANCE){
      safe = false;
    }
  }
  return safe;
}

LANE Road::lane_change_available(Car& car){
  LANE car_lane = car.get_lane();
  LANE target_lane = car_lane;

  if (car_lane == LANE::LEFT || car_lane == LANE::RIGHT) {
    if (this->free_lane(car, LANE::MIDDLE)) {
      target_lane = LANE::MIDDLE;
    }
  } else {
    if (this->free_lane(car, LANE::LEFT)) {
      target_lane = LANE::LEFT;
    } else if (this->free_lane(car, LANE::RIGHT)) {
      target_lane = LANE::RIGHT;
    }
  }
  return target_lane;
}

bool Road::free_lane(Car& car, LANE lane){
  vector<Car> target_lane = this->get_lane_cars(lane);
  bool is_free = true;
  for (int i = 0; i < target_lane.size(); i++) {
    double distance = std::abs(target_lane[i].get_s() - car.get_s());
    if(distance < GUARD_DISTANCE){
      is_free = false;
    }
  }
  return is_free;
}