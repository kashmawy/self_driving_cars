#include "car.h"

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
    if (this->d < 4.0) {
        return LANE::LEFT;
    } else if (this->d >= 4.0 && this->d < 8.0) {
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