#include "map.h"

using namespace std;
using namespace tk;

Map::Map() {}

Map::~Map() {}

Map::Map(string map_file_) {
    vector<double> map_waypoints_x;
    vector<double> map_waypoints_y;
    vector<double> map_waypoints_s;
    vector<double> map_waypoints_dx;
    vector<double> map_waypoints_dy;

    ifstream in_map_(map_file_.c_str(), ifstream::in);

    string line;
    while (getline(in_map_, line)) {
        istringstream iss(line);
        double x;
        double y;
        float s;
        float d_x;
        float d_y;

        iss >> x;
        iss >> y;
        iss >> s;
        iss >> d_x;
        iss >> d_y;

        map_waypoints_x.push_back(x);
        map_waypoints_y.push_back(y);
        map_waypoints_s.push_back(s);
        map_waypoints_dx.push_back(d_x);
        map_waypoints_dy.push_back(d_y);
    }

    this->wp_spline_x.set_points(map_waypoints_s, map_waypoints_x);
    this->wp_spline_y.set_points(map_waypoints_s, map_waypoints_y);
    this->wp_spline_dx.set_points(map_waypoints_s, map_waypoints_dx);
    this->wp_spline_dy.set_points(map_waypoints_s, map_waypoints_dy);
}

vector<double> Map::getXY(double s, double d) {
    double wp_x, wp_y, wp_dx, wp_dy, next_x, next_y;

    wp_x = this->wp_spline_x(s);
    wp_y = this->wp_spline_y(s);
    wp_dx = this->wp_spline_dx(s);
    wp_dy = this->wp_spline_dy(s);

    next_x = wp_x + wp_dx * d;
    next_y = wp_y + wp_dy * d;

    return {next_x, next_y};
}