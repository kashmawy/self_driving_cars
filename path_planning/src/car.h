#ifndef CAR_H
#define CAR_H

#include <vector>
#include "constants.h"

using namespace std;

class Car {
    public:
        Car();
        Car(int id, double x, double y, double v, double s, double d);
        ~Car();
        LANE get_lane();
        double get_x();
        double get_y();
        double get_v();
        double get_s();
        double get_d();
        double get_yaw();
        void set_previous_s(vector<double> previous_s);
        void set_previous_d(vector<double> previous_d);
        vector<double> prev_s();
        vector<double> prev_d();

        void update_values(double x, double y, double v, double s, double d, double yaw);

    private:
        int id;
        double x, y, v, s, d, yaw;
        vector<double> previous_s;
        vector<double> previous_d;
};

#endif