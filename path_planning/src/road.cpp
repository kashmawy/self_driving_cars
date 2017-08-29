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