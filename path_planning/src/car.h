#ifndef CAR_H
#define CAR_H

#include "constants.h"

class Car {
    public:
        Car();
        Car(int id, double x, double y, double v, double s, double d);
        ~Car();
        LANE get_lane();
        void update_values(double x, double y, double v, double s, double d, double yaw);

        int id;
        double x, y, v, s, d, yaw;
};

#endif