#include "utils.h"

LANE get_lane(double d) {
  if (d < LEFT_LANE_D_END) {
    return LANE::LEFT;
  }
  else if ((d >= LEFT_LANE_D_END) && (d < MIDDLE_LANE_D_END)) {
    return LANE::MIDDLE;
  }
  else {
    return LANE::RIGHT;
  }
}

double get_lane_d_from_d(double d) {
  if (d < LEFT_LANE_D_END) {
    return LEFT_LANE_D;
  }
  else if (d >= LEFT_LANE_D_END && d < MIDDLE_LANE_D_END) {
    return MIDDLE_LANE_D;
  }
  else {
    return RIGHT_LANE_D;
  }
}

double get_lane_d_from_lane(LANE lane) {
  if (lane == LANE::LEFT) {
    return LEFT_LANE_D;
  }
  else if (lane == LANE::MIDDLE) {
    return MIDDLE_LANE_D;
  }
  else {
    return RIGHT_LANE_D;
  }
}