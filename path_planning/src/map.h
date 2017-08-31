#ifndef MAP_H
#define MAP_H

#include "spline.h"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>

using namespace std;
using namespace tk;

class Map {
    public:
        Map();
        Map(string in_map_file_);
        ~Map();

        vector<double> getXY(double s, double d);

        spline spline_x;
        spline spline_y;
        spline spline_dx;
        spline spline_dy;
};

#endif