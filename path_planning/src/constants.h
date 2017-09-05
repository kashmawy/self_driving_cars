#ifndef CONSTANTS_H
#define CONSTANTS_H

enum class LANE { LEFT, MIDDLE, RIGHT };
enum class STATE { START, KEEP_LANE, MOVE_LEFT, MOVE_RIGHT };
const double AT = 0.02;  // s
const double TRACK_DISTANCE = 6945.564;
const double ROAD_WIDTH = 12.0;  // in meters
const double POINTS = 50;
const double FRONT_SAFE_DISTANCE = 40.0; // m
const double BACK_SAFE_DISTANCE = -20.0; // m
const double CYCLES = 2;
const double LEFT_LANE_D_START = 0;
const double LEFT_LANE_D = 2.0;
const double LEFT_LANE_D_END = 4.0;
const double MIDDLE_LANE_D = 6.0;
const double MIDDLE_LANE_D_END = 8.0;
const double RIGHT_LANE_D = 10.0;
const double RIGHT_LANE_D_END = 12.0;
const double SPEED_LIMIT = 20.0; //m/s

#endif