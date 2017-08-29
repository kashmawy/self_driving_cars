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

        spline wp_spline_x;
        spline wp_spline_y;
        spline wp_spline_dx;
        spline wp_spline_dy;
};

#endif