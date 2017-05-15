# Overview
Localization using 2d particle filter.

#### Project Info

The project implements a 2d particle filter in C++ to localize the location of a robot.
The robot has:

 1. A map of this location.
 2. Noisy GPS estimate of its initial location.
 3. Noisy sensor and control data.
 4. At each time step the filter will also get observation and control data.

## Building and running code
Once you have this repository on your machine, `cd` into the repository's root directory and run the following commands from the command line:

```
> ./clean.sh
> ./build.sh
> ./run.sh
```

> **NOTE**
> If you get any `command not found` problems, you will have to install 
> the associated dependencies (for example, 
> [cmake](https://cmake.org/install/))

If everything worked you should see something like the following output:

Time step: 2444
Cumulative mean weighted error: x .1 y .1 yaw .02
Runtime (sec): 38.187226
Success! Your particle filter passed!

```
Otherwise you might get
.
.
.
Time step: 100
Cumulative mean weighted error: x 39.8926 y 9.60949 yaw 0.198841
Your x error, 39.8926 is larger than the maximum allowable error, 1
```

which means that the error has been large and the filter did not pass the test.

## Inputs to the Particle Filter

You can find the inputs to the particle filter in the `data` directory.

#### The Map*
`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian coordinate system. Each row has three columns
1. x position
2. y position
3. landmark id

> * Map data provided by 3D Mapping Solutions GmbH.


#### Control Data
`control_data.txt` contains rows of control data. Each row corresponds to the control data for the corresponding time step. The two columns represent
1. vehicle speed (in meters per second)
2. vehicle yaw rate (in radians per second)

#### Observation Data
The `observation` directory includes around 2000 files. Each file is numbered according to the timestep in which that observation takes place.

These files contain observation data for all "observable" landmarks. Here observable means the landmark is sufficiently close to the vehicle. Each row in these files corresponds to a single landmark. The two columns represent:
1. x distance to the landmark in meters (right is positive) RELATIVE TO THE VEHICLE.
2. y distance to the landmark in meters (forward is positive) RELATIVE TO THE VEHICLE.

> **NOTE**
> The vehicle's coordinate system is NOT the map coordinate system. The code handles this transformation.

# Directory Structure
The directory structure of this repository is as follows:

```
root
|   build.sh
|   clean.sh
|   CMakeLists.txt
|   README.md
|   run.sh
|
|___data
|   |   control_data.txt
|   |   gt_data.txt
|   |   map_data.txt
|   |
|   |___observation
|       |   observations_000001.txt
|       |   ...
|       |   observations_002444.txt
|
|___src
    |   helper_functions.h
    |   main.cpp
    |   map.h
    |   particle_filter.cpp
    |   particle_filter.h
```


# Implementing the Particle Filter

The code contains the main module and the particle filter.
The main module does the following:

1. Reads the map data, position data and ground truth.
2. Iterate over the time steps and for each time step, do the following:
3. Read the landmark observations
4. Initialize the particle filter if this is the first time, otherwise predict the vehicle next state (given the delta_t, previous velocity and yaw rate)
5. update the particles weights and resample

The particle filter module does the following:

1. Initialize sets x, y, theta to be the given g, y and theta using a normal distribution
2. Predicts takes the delta_t, x and y standard deviation, velocity and yaw rate and does the following:

   Iterate over each particle
   For each particle, predict theta to be previous theta + yaw_rate * delta_t
   predict x to be previous x + velocity/yaw_rate * (sin(theta_prediction) - sin(previous theta))
   predict y to be previous y + velocity/yaw_rate * (cos(theta_prediction) - cos(previous theta))

   Set current x to be predicted x with normal distribution
   Set current y to be predicted y with normal distribution
   Set current theta to be predicted theta with normal distribution
   Set current weight to be 1.0

3. Update weights does the following:

  Iterate over each particle and for each particle
  Transform the observations from map coordinate to particle coordinate by doing the following:

    Creating a transformed observation for each observation

    The transformed observation have the same id

    The transformed observation x coordinate = transformed observation x coordinate * cos(-particle.theta) + transformed observation y coordinate * sin(-particle.theta) + particle.x

    The transformed observation y coordinate = -transformed observation x coordinate * sin(-particle.theta) + current observation y coordinate * cos(-particle.theta) + particle.y

  Iterate over transformed observations and for each transformed observation, get the corresponding landmark from the map landmarks that has the shortest distance
  Do bivariate normal distribution given the transformed observation x and y coordinates, the landmark x and y coordinates, and the standard deviation of the landmark
  Update the particle weight to be equal bivariate normal distribution over all the transformed observation and the corresponding landmark multiplied

  In the end normalize all the particle weights

4. Resample which does the following:

  Get a discrete distribution given all the weights of all the particles


In the end, the code looks for the following:

1. **Accuracy**: particle filter should localize vehicle position and yaw to within the values specified in the parameters `max_translation_error` (maximum allowed error in x or y) and `max_yaw_error` in `src/main.cpp`.
2. **Performance**: particle filter should complete execution within the time specified by `max_runtime` in `src/main.cpp`.
