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

bool Road::is_free_lane(Car& car, LANE lane){
  vector<Car> r_car_lane = this->get_lane_cars(lane);
  for (int i = 0; i < r_car_lane.size(); i++) {
    double distance = r_car_lane[i].get_s() - car.get_s();
    if(distance > BACK_SAFE_DISTANCE && distance < FRONT_SAFE_DISTANCE){
      return false;
    }
  }

  return true;
}

LANE Road::get_free_lane(Car& car){
  LANE car_lane = car.get_lane();

  if (car_lane == LANE::LEFT || car_lane == LANE::RIGHT) {
    if (this->is_free_lane(car, LANE::MIDDLE)) {
      return LANE::MIDDLE;
    }
  } else if (this->is_free_lane(car, LANE::RIGHT)) {
      return LANE::RIGHT;
  } else if (this->is_free_lane(car, LANE::LEFT)) {
      return LANE::LEFT;
  }

  return car_lane;
}