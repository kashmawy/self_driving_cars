#ifndef ROAD_H
#define ROAD_H

#include <string>
#include <vector>
#include <cmath>
#include "car.h"
#include "constants.h"

using namespace std;

class Road {
    public:
        Road();
        ~Road();

        void update_road(vector<Car> left_lane, vector<Car> middle_lane, vector<Car> right_lane);
        vector<Car> get_lane_cars(LANE lane);
        bool safe_lane(Car& car, LANE lane);
        LANE lane_change_available(Car& car);
        bool free_lane(Car& car, LANE lane);

        vector<Car> left_lane;
        vector<Car> middle_lane;
        vector<Car> right_lane;
};

#endif