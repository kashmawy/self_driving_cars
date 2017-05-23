# Model Predictive Controller Project

---


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
* [uWebSockets](https://github.com/uWebSockets/uWebSockets) == 0.14, but the master branch will probably work just fine
  * Follow the instructions in the [uWebSockets README](https://github.com/uWebSockets/uWebSockets/blob/master/README.md) to get setup for your platform. You can download the zip of the appropriate version from the [releases page](https://github.com/uWebSockets/uWebSockets/releases). Here's a link to the [v0.14 zip](https://github.com/uWebSockets/uWebSockets/archive/v0.14.0.zip).
  * If you have MacOS and have [Homebrew](https://brew.sh/) installed you can just run the ./install-mac.sh script to install this.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * Mac: `brew install ipopt --with-openblas`
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/) or the [Github releases](https://github.com/coin-or/Ipopt/releases) page.
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/CarND-MPC-Project/releases).


## Basic Build Instructions


1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.


## Code Structure

Main does the following:

1. As the simulator is running, it gets a 'telemetry' event which has the following information:

  a. (ptsx, ptsy) which are the map coordinates
  b. (px, py) which are the vehicle coordinates
  c. psi which is the angle
  d. psi_unity
  e. v which is the speed

2. Create ptsxv and ptsyv which are the map coordinates
3. Convert them into vehicle space coordinates by doing the following calculation:

  a. x = ptsxv - px
  b. y = ptsyv - py
  c. new_x_coordinate = x * cos(psi) + y * sin(psi)
  d. new_y_coordinate = -x * sin(psi) + y * cos(psi)

4. Fit ptsxv and ptsyv into 3rd degree polynomial and get the coefficients.
5. Get cross track error by evaluating the polynomial with the coefficients from the previous step and at x = 0 and multiply it be -1.
6. Calculate psi error (epsi) by evaluating atan(coefficients[1]) and multiplying it by -1.
7. Create the current state from (px, py, psi, v, cte, epsi)
8. Solve given the current state and the constraints (handled by the MPC module)
9. Get the steer value and the throttle value
10. Display the MPC predicted trajectory and the reference points

MPC module does the following:

1. Get the current state from the input (x, y, psi, v, cte, epsi)
2. Define the number of constraints to be 6
3. Set the upper bound and lower bound for:

    a. all variables (big negative number, big positive number)
    b. delta to be [-25 degrees to 25 degrees]
    c. accelerator to be [-1 to 1]

4. Compute the solution from the constraints using the following:

    a. Define the cost to accumulate the following:

        1. power(velocity difference to reference velocity of 50, 2) => Will keep velocity close to 0
        2. power(cte difference to reference cte of 0, 2) => Will keep CTE close to 0
        3. 2 * power(epsi difference to reference epsi of 0, 2) => Will keep EPSI close to 0
        4. power(delta, 2)
        5. power(acceleration, 2)
        6. 200 * power(delta difference, 2) => Will make changes in delta happen more slowly and smoothly
        7. 10 * power(acceleration difference, 2) => Will make changes in acceleration happen more slowly and smoothly

    b. Copy over the state for the next N times
    c. Create all the constraints of the following:

        1. X constraint: x1 - (x0 + v0 * cos(psi0) * dt)
        2. Y constraint: y1 - (y0 + v0 * sin(psi0) * dt)
        3. PSI Constraint: psi1 - (psi0 + v0 * delta0 / Lf * dt)
        4. V Constraint: v1 - (v0 + a0 * dt)
        5. CTE Constraint: cte1 - ((f0 - y0) + (v0 * sin(epsi0) * dt))
        6. EPSI Constraint: epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt)

        where f0 = coeffs[0] + (coeffs[1] * x0) + (coeffs[2] * pow(x0,2)) + (coeffs[3] * pow(x0,3))
        and psides0 = atan(coeffs[1] + (2 * coeffs[2] * x0) + (3 * coeffs[3]* pow(x0,2) ))


5. Get the delta and acceleration from the solution

## Simulation

The following shows simulation using this Model Predictive Controller.
The green line represents the predicted path while the yellow line represents the ground truth.

[![Simulation using MPC](https://img.youtube.com/vi/NxKmWrKG7eY/0.jpg)](https://youtu.be/NxKmWrKG7eY)

## Credit

Tips have been followed from https://github.com/hortovanyi/CarND-MPC-Project