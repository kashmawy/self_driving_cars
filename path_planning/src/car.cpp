#include "car.h"
#include "constants.h"


Car::Car() {
    this->id = -1;
}

Car::Car(int id, double x, double y, double v, double s, double d) {
    this->id = id;
    this->x = x;
    this->y = y;
    this->v = v;
    this->s = s;
    this->d = d;
}

Car::~Car() {
}

LANE Car::get_lane() {
    if (this->d < LEFT_LANE_D_END) {
        return LANE::LEFT;
    } else if (this->d >= LEFT_LANE_D_END && this->d < MIDDLE_LANE_D_END) {
        return LANE::MIDDLE;
    } else {
        return LANE::RIGHT;
    }
}

void Car::update_values(double x, double y, double v, double s, double d, double yaw) {
    this->x = x;
    this->y = y;
    this->v = v;
    this->s = s;
    this->d = d;
    this->yaw = yaw;
}

double Car::get_x() { return this->x; }
double Car::get_y() { return this->y; }
double Car::get_v() { return this->v; }
double Car::get_s() { return this->s; }
double Car::get_d() { return this->d; }
double Car::get_yaw() { return this->yaw; }
void Car::set_previous_s(vector<double> previous_s){
  this->previous_s = previous_s;
}

void Car::set_previous_d(vector<double> previous_d){
  this->previous_d = previous_d;
}

vector<double> Car::prev_s(){
  return this->previous_s;
}

vector<double> Car::prev_d(){
  return this->previous_d;
}