Localization project using Laser and Radar data by unscented kalman filter
==========================================================================

## Dependencies

* cmake >= v3.5
* make >= v4.1
* gcc/g++ >= v5.4

## How to build and run

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./UnscentedKF path/to/input.txt path/to/output.txt`. You can find
   some sample inputs in 'data/'.
    - eg. `./UnscentedKF ../data/obj_pose-laser-radar-synthetic-input.txt`

## Code Structure and Explanation

The code is structured into a main component and a ukf component.
The main component reads the measurement from an input file and calls ProcessMeasurement with that line on ukf.

The ukf component initializes the following variables:

1. X the state vector. [5x1 vector]
2. P the covriance matrix. [5x5 matrix]
3. std_a, the standard deviation longitudinal acceleration.
4. std_yawdd, the noise standard deviation yaw acceleration.
5. std_laser, the laser measurement noise standard deviation position.
6. std_radar, the radar measurement noise standard deviation radius.
7. std_radphi, the radar measurement noise standard deviation angle.
8. std_radrd, the radar measurement noise standard deviation radius change.
9. n_x, the state dimension.
10. n_aug, the augemented state dimension.
11. n_z_radar, Radar Z dimension
12. n_z_laser, Laser Z dimension
13. lambda, 3 - n_x

The ukf component does the following each time ProcessMeasurement is called:

1. If this is the first measurement, then:

  If it is a radar measurement, then extract rho (first value) and phi (second value) and compute px (rho * cos(phi)) and py (rho * sin(phi)) and set X to px and py.

  If it is a laser measurement, set X to first value (which is px) and second value (which is py).

  Record the timestamp in previous_timestamp and mark it such that we know that we have processed a measurement already.

2. For all subsequent measurement, if it is a radar then call Predict passing dt (microseconds difference between measurement timestamp and current timestamp) and then call UpdateRadar.
3. If it is a laser measurement, then call Predict passing dt and then call UdateLaser.

Predict does the following given dt:

1. Calculate Sigma point augmented
2. Predict Sigma Point
3. Predict State Mean
4. Predict State Covariance Matrix

UpdateRadar does the following given the current measurement and the sigma point prediction:

1. Create Zsig matrix from the sigma point prediction
2. Calculate the mean predicted measurement
3. Calculate S from Zsig and weights
4. Calculate R from the standard deviation of the measurement
5. Calculate Tc from weights, z_diff and x_diff
6. Calculate K from Tc and S
7. Calculate X and P from the previous

UpdateLaser is very simmiliar.

## Credit

Some of the work here was inspired by [here](https://github.com/Valtgun/CarND-Unscented-Kalman-Filter-Project)