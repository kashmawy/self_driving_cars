#ifndef CONSTANTS_H
#define CONSTANTS_H

enum class LANE { LEFT, MIDDLE, RIGHT };

enum class STATE { START, KEEP_LANE, MOVE_LEFT, MOVE_RIGHT };

// s
const double AT = 0.02;

const double TRACK_DISTANCE = 6945.554;

// in meters
const double ROAD_WIDTH = 12.0;

#endif