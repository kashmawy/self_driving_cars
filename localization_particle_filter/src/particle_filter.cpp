/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <math.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 1000;

    default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	particles.resize(num_particles);
	weights.resize(num_particles, 1.0);

	for(int i = 0; i < num_particles; ++i) {
	    particles[i].id = i;
	    particles[i].x = dist_x(gen);
	    particles[i].y = dist_y(gen);
	    particles[i].theta = dist_theta(gen);
	    particles[i].weight = 1.0;
	}
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    std::default_random_engine gen;

    for(int i=0; i < num_particles;i++) {
        Particle &p = particles[i];

        double x_pred;
        double y_pred;
        double theta_pred;

        theta_pred = p.theta + yaw_rate*delta_t;
        x_pred = p.x + velocity/yaw_rate*(sin(theta_pred)-sin(p.theta));
        y_pred = p.y - velocity/yaw_rate*(cos(theta_pred)-cos(p.theta));

        std::normal_distribution<double> dist_x(x_pred,std_pos[0]);
        std::normal_distribution<double> dist_y(y_pred,std_pos[1]);
        std::normal_distribution<double> dist_theta(theta_pred,std_pos[2]);

        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
    }
}

Map::single_landmark_s ParticleFilter::getCorrespondingLandmark(LandmarkObs observation, Map landmarks) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	int landmark_index = -1;
    double min_distance = numeric_limits<double>::max();
    for (int i = 0; i < landmarks.landmark_list.size(); ++i) {
        Map::single_landmark_s lm = landmarks.landmark_list[i];
        double cur_distance = dist(observation.x, observation.y, lm.x_f, lm.y_f);
        if (cur_distance < min_distance) {
                min_distance = cur_distance;
                landmark_index = i;
        }
    }

    return landmarks.landmark_list[landmark_index];
}

vector<LandmarkObs> ParticleFilter::transformObservations(Particle particle, vector<LandmarkObs> observations) {
    // http://planning.cs.uiuc.edu/node99.html
    // rotation and translation from map coordinates [y up, x right] to vehicle coordinates [x up, y left]
    vector<LandmarkObs> transformed_observations;
    for (int i = 0; i < observations.size(); ++i) {
        LandmarkObs current_observation = observations[i];
        LandmarkObs transformed_observation;
        transformed_observation.id = current_observation.id;
        transformed_observation.x = current_observation.x * cos(-particle.theta) + current_observation.y * sin(-particle.theta) + particle.x;
        transformed_observation.y = -current_observation.x * sin(-particle.theta) + current_observation.y * cos(-particle.theta) + particle.y;
        transformed_observations.push_back(transformed_observation);
    }

    return transformed_observations;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

    for (int i = 0; i < num_particles; ++i) {
        Particle& particle = particles[i];
        vector<LandmarkObs> transformed_observations = transformObservations(particle, observations);

        double total_prob = 1.0;
        for (int j = 0; j < transformed_observations.size(); ++j) {
            LandmarkObs current_observation = transformed_observations[j];
            Map::single_landmark_s current_landmark = getCorrespondingLandmark(current_observation, map_landmarks);

            total_prob *= bivariate_normal(current_observation.x,
                                           current_observation.y,
                                           current_landmark.x_f,
                                           current_landmark.y_f,
                                           std_landmark[0],
                                           std_landmark[1]);
        }

        particle.weight = total_prob;
        weights[i] = total_prob;
    }

    double wg_sum = 0.;
    for(int i = 0;i < particles.size(); i++) {
        wg_sum += particles[i].weight;
    }

    for(int i = 0; i < particles.size(); i++) {
        particles[i].weight /= wg_sum;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    std::vector<double> weights(particles.size());
    for(int i=0;i<particles.size();i++) {
        weights[i] = particles[i].weight;
    }
    discrete_distribution<int> index(weights.begin(), weights.end());
    vector<Particle> resampled_particles;

    for (int counter = 0; counter < particles.size(); counter++) {
        int weighted_index = index(gen);
        resampled_particles.push_back(particles[index(gen)]);
    }

    particles = resampled_particles;

    for(int i=0;i < particles.size();i++){
        particles[i].weight = 1.0;
    }
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
