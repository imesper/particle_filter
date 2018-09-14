/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <array>
#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;

    // This line creates a normal (Gaussian) distribution for x.
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {
        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1;
        particles.push_back(particle);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    for (unsigned int i = 0; i < particles.size(); ++i) {
        double x_pred, y_pred, theta_pred = 0;
        if(fabs(yaw_rate) < 0.000001){
            x_pred = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            y_pred = particles[i].y + velocity *delta_t * sin(particles[i].theta);
        } else {
            x_pred = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            y_pred = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta +yaw_rate*delta_t ));
            theta_pred = particles[i].theta + yaw_rate * delta_t;
        }
        // This line creates a normal (Gaussian) distribution for x.
        normal_distribution<double> dist_x(x_pred, std_pos[0]);
        normal_distribution<double> dist_y(y_pred, std_pos[1]);
        normal_distribution<double> dist_theta(theta_pred, std_pos[2]);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    // calculate normalization term
    double gauss_norm = 1.0/(2 * M_PI * std_landmark[0] * std_landmark[1]);

    for (unsigned int i = 0; i < particles.size(); ++i) {

        double weights = 1;

        for(auto &obs: observations){
            // Transform to MAP Coordinate System
            LandmarkObs observation = obs;
            observation.x = transform_x(particles[i].theta, obs.x, obs.y, particles[i].x);
            observation.y = transform_y(particles[i].theta, obs.x, obs.y, particles[i].y);

            double small_dist = numeric_limits<double>::max();
            unsigned int index = 0;
            for(auto landmarks: map_landmarks.landmark_list){
                // Check if landmark is in sensor range
                if(fabs(static_cast<double>(landmarks.x_f) - particles[i].x) <= sensor_range &&
                        fabs(static_cast<double>(landmarks.y_f) - particles[i].y) <= sensor_range) {

                    double c_dist = dist(static_cast<double>(landmarks.x_f), static_cast<double>(landmarks.y_f), observation.x, observation.y);
                    if(c_dist < small_dist){
                        small_dist = c_dist;
                        index = static_cast<unsigned int>(landmarks.id_i);
                    }
                }
            }
            // index is always +1, so let's correct this
            index--;
            double mu_x = static_cast<double>(map_landmarks.landmark_list[index].x_f);
            double mu_y = static_cast<double>(map_landmarks.landmark_list[index].y_f);

            // calculate exponent
            double sub_x =  mu_x - observation.x;
            double sub_y =  mu_y - observation.y;

            double exponent = (pow(sub_x,2)/(2 * pow(std_landmark[0],2))) + (pow(sub_y,2)/(2 * pow(std_landmark[1],2)));

            // calculate weight using normalization terms and exponent
            double w = gauss_norm * exp(-exponent);
            weights *= w;
        }
        particles[i].weight = weights;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::vector<double> weights;

    for(size_t i=0; i < particles.size(); ++i) {
        weights.push_back(particles[i].weight);
    }

    // Normalize Weightsto use a discrete distribution
    double total_weight = 0.0;

    for (unsigned int i=0; i < weights.size(); i++)
    {
        total_weight += weights[i];
    }

    for (unsigned int i=0; i < weights.size(); i++)
    {
        weights[i] /= total_weight;
        particles[i].weight /= total_weight;
    }

    std::discrete_distribution<size_t> d(weights.begin(), weights.end());
    vector<Particle> new_particles;

    for(unsigned int n=0; n < particles.size(); ++n) {
        new_particles.push_back(particles[d(gen)]);
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
