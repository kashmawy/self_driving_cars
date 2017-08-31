#include "map.h"

using namespace std;
using namespace tk;

Map::Map() {}

Map::~Map() {}

Map::Map(string map_file_) {
    vector<double> map_x, map_y, map_s, map_dx, map_dy;

    ifstream in_map_(map_file_.c_str(), ifstream::in);

    string line;
    while (getline(in_map_, line)) {
        istringstream iss(line);
        double x, y;
        float s, d_x, d_y;

        iss >> x;
        iss >> y;
        iss >> s;
        iss >> d_x;
        iss >> d_y;

        map_x.push_back(x);
        map_y.push_back(y);
        map_s.push_back(s);
        map_dx.push_back(d_x);
        map_dy.push_back(d_y);
    }

    this->spline_x.set_points(map_s, map_x);
    this->spline_y.set_points(map_s, map_y);
    this->spline_dx.set_points(map_s, map_dx);
    this->spline_dy.set_points(map_s, map_dy);
}

vector<double> Map::getXY(double s, double d) {
    double wp_x, wp_y, wp_dx, wp_dy, next_x, next_y;

    wp_x = this->spline_x(s);
    wp_y = this->spline_y(s);
    wp_dx = this->spline_dx(s);
    wp_dy = this->spline_dy(s);

    next_x = wp_x + wp_dx * d;
    next_y = wp_y + wp_dy * d;

    return {next_x, next_y};
}