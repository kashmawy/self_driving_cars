# PID Controller

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
* [uWebSockets](https://github.com/uWebSockets/uWebSockets) == 0.13, but the master branch will probably work just fine
  * Follow the instructions in the [uWebSockets README](https://github.com/uWebSockets/uWebSockets/blob/master/README.md) to get setup for your platform. You can download the zip of the appropriate version from the [releases page](https://github.com/uWebSockets/uWebSockets/releases). Here's a link to the [v0.13 zip](https://github.com/uWebSockets/uWebSockets/archive/v0.13.0.zip).
  * If you run OSX and have homebrew installed you can just run the ./install-mac.sh script to install this
* Simulator. You can download these from the [project intro page](https://github.com/udacity/CarND-PID-Control-Project/releases) in the classroom.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`.

## PID Controller

This project creates a PID controller. The following are the elements of the PID controller:

1. Proportional to Cross Track Error (CTE): proportional to - alpha1 * CTE in reference to where we want to be. The P component makes sure that we approach the Y value of where we want to be.
2. Integral to Cross Track Error (CTE): proportional to - alpha2 * [summation of all CTE at all time frames in reference to where we want to be]. The I component is used to guard against the drift error.
3. Diffrential to Cross Track Error (CTE): proportional to - alpha3 * [diffrence in CTE in reference to where we want to be]. The D component ensures that we have less oscilliations when reaching the Y value of where we want to be.

# PID values

The alpha1, alpha2 and alpha3 values for PID were chosen by manually testing them with the simulated track and seeing the effect and changing them accordingly.

# Results

The following are the results of using the controller with track1:

1. PID Controller

[![The PID controller completes the track successfully](https://img.youtube.com/vi/7OSPnfSJog8/0.jpg)](https://youtu.be/7OSPnfSJog8)

2. PD Controller (No I)

The I component mainly tries to compoensate for drift error which is negligible in this case.

[![A controller without I also completes the track successfully, which means that the drift error is negligible.](https://img.youtube.com/vi/YW43OYxFKzc/0.jpg)](https://youtu.be/YW43OYxFKzc)

3. ID Controller (No P)

The P component mainly tries to reduce the CTE error, since it is ignored in this case, the vehicle fails to correct its position to minimize CTE and be where it should be and the track run fails.

[![A controller without P does not complete the track successfully](https://img.youtube.com/vi/k8yrqYgbTqo/0.jpg)](https://youtu.be/k8yrqYgbTqo)

4. PI Controller (No D)

The D component mainly tries to reduce the diff in CTE in order to be able to reach the desired location without overshooting in both directions as it minimizes the CTE. This is why this track run fails.

[![A controller without the D does not complete the track successfully](https://img.youtube.com/vi/DdLDQ8rNstQ/0.jpg)](https://youtu.be/DdLDQ8rNstQ)
