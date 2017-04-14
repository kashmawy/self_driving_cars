Localization project using Laser and Radar data by extended kalman filter
=========================================================================


## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## How to build and run

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make` 
   * On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./ExtendedKF path/to/input.txt path/to/output.txt`. You can find
   some sample inputs in 'data/'.
    - eg. `./ExtendedKF ../data/sample-laser-radar-measurement-data-1.txt output.txt`


## Code Structure and Explanation

The code is organized into main component, FusionEKF component and kalman_filter component.
Data structures were created to be used which are ground_truth_package and measurement_package.

The main component reads the input file, line by line and calls ProcessMeasurement on FusionEKF component.

The FusionEKF component is responsible for setting the following values on initialization:

1. R for laser is the measurement covariance matrix for Laser.
2. R for radar is the measurement covariance matrix for Radar.
3. H for laser is the measurement matrix, which projects your belief about the object's current state into the measurement space of the sensor.
4. P is the state covariance matrix, which contains information about the uncertainity of the object's position and velocity. (like standard deviation)
5. F is a matrix that when multiplied with X predicts where the object will be after delta t.
6. Q is the process covariance matrix.

On each ProcessMeasurement, it does the following:

1. If this was the first value processed:

   Then if it is a Radar measurement then extract (range, bearing and rho velocity) and calculate x, y, vx, and vy from these values.

   If it is a laser measurement, then extract x, y, and set vx and vy to 0 and 0.

   Check if x and y values are less than a small value, if so, set them to those values.
   Set a record to indicate that the first measurement was record and record its timestamp.

2. If this was not the first value processed
   Calculate the time delta between the current measurement and the previous measurement in microseconds.
   Set F to X
   Set Q to Y

3. Call Predict on kalman_filter component, which does the following calculation.
   X = F * X
   P = F * P * Ftranspose + Q

4. Call Update on kalman_filter component, which does the following:

   If the measurement is a radar measurement, then call UpdateEKF which does the following:
   
     Set Hj to be Jacobian from X.
     Calculate range, bearing and rho velocity from X.
     Set h to be range, bearing and rho velocity.
     Set y to be z (current measurement) - h.

     Call UpdateCommon with Hj, R_radar and y, which does the following:

     Ht = H.tranpose()
     S = H * P * Ht + R
     K = P * Ht * S.inverse()

     X = X + (K * y)
     I = Identify matrix of same size as X
     P = (I - K * H) * P

   If the measurement is a laser measurement, then call Update which does the following:
     Set y to be z - H * x
     Call UpdateCommon with H, R_laser and y which was explained above.

The process described above describes the process we go with each measurement (laser or radar).

At the end we calculate the root mean squared error and compare it with the actual value to verify that we produced a value of high accuracy.

