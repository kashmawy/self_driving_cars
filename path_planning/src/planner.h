#ifndef PLANNER_H
#define PLANNER_H

#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/Dense"
#include <vector>
#include <iostream>

#include "car.h"
#include "map.h"
#include "road.h"
#include "constants.h"
#include "utils.h"

using namespace std;

class Planner {

    public:
        Planner();
        ~Planner();

        vector<double> JMT(vector<double> start, vector<double> end, double T);
        void create_new_trajectory_points(Map& map, vector<vector<double>>& trajectory);
        void create_trajectory(Map& map, Road& road, Car& car, vector<vector<double>>& trajectory);

        void update_state(LANE current_lane, LANE target_lane);

        void apply_action(Car& car, LANE current_lane, LANE target_lane);
        void start_car(Car& car);
        void stay_in_lane(Car& car);
        void decrease_speed(Car& car);
        void change_lane(Car& car, LANE target_lane);


        int n;
        STATE state;
        vector<double> start_s;
        vector<double> end_s;
        vector<double> start_d;
        vector<double> end_d;
        bool new_path;
};

#endif