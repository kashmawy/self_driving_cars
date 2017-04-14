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
3. H for laser is the measurement matrix, which projects our belief about the object's current state into the measurement space of the sensor.
4. P is the state covariance matrix, which contains information about the uncertainity of the object's position and velocity, like standard deviation.
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

   Set F to the following:

       [ 1 0 dt 0 ]
       [ 0 1 0 dt ]
       [ 0 0 1 0  ]
       [ 0 0 0 1  ]

   Set Q to the following:

        [ dt^4/4*noise_ax, 0, dt^3/2*noise_ax, 0 ]
        [ 0, dt^4/4*noise_ay, 0, dt^3/2*noise_ay ]
        [ dt^3/2*noise_ax, 0, dt^2*noise_ax, 0 ]
        [ 0, dt^3/2*noise_ay, 0, dt^2*noise_ay ]

3. Call Predict on kalman_filter component, which does the following calculation.

   X = F * X [Calculate where is the next X]

   P = F * P * Ftranspose + Q [Calculate the next P]

4. Call Update on kalman_filter component, which does the following:

   If the measurement is a radar measurement, then call UpdateEKF which does the following:

     Set Hj to be Jacobian from X. [A linear H for radar]

     Calculate range [radial distance from origin], bearing [angle between p which is where the object is moving toward and x] and rho velocity [range rate] from X.

     Set h to be range, bearing and rho velocity.

     Set y to be z (current measurement) - h.

     Call UpdateCommon with Hj, R_radar and y, which does the following:

     Ht = H.tranpose()

     S = H * P * Ht + R []

     K = P * Ht * S.inverse() [Kalman filter gain, combines the uncertainty of where we think we are P with the uncertainty of our sensor measurement R]

     X = X + (K * y) (Calculate the next X)

     I = Identify matrix of same size as X

     P = (I - K * H) * P (Calculate the next P)

   If the measurement is a laser measurement, then call Update which does the following:

     Set y to be z - H * x

     Call UpdateCommon with H, R_laser and y which was explained above.

The process described above describes the process we go with each measurement (laser or radar).

At the end we calculate the root mean squared error and compare it with the actual value to verify that we produced a value of high accuracy.


## Results

For the first shape, the root mean squared for px, py, vx, vy was: [0.0651648, 0.0605379, 0.533212, 0.544193].
You can see the details in the following figure:


![8 Shape](visualizations/ekf_8_shape.png)

The yellow dots represent the measurement.

The green line represent the ground truth while the blue line represent where the extended kalman filter thinks the object is.

The y axis is the position along the y axis, and the x axis is the position along the x-axis.

For the second shape, the root mean squared for px, py, vx, vy was: [0.187718, 0.192202, 0.474744, 0.830695]
You can see the details in the following figure:

![N shape](visualizations/ekf_n_shape.png)

## Credit

Some of the work here was inspired by [here](https://github.com/NikolasEnt/Extended-Kalman-Filter)