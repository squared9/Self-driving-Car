/*
 * particle_filter.cpp
 * Implementation (c) 2017 squared9
 * 1st run of SDC ND T2
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

#include "particle_filter.h"

#define M_2PI 2 * M_PI
#define EPSILON 1E-4
//#define IS_DEBUG

using namespace std;

static default_random_engine rnd; // generator needed for random numbers

/**
 * Initializes particles around assumed position with noise
 * @param x assumed x
 * @param y assumed y
 * @param theta assumed orientation
 * @param std array of standard deviations for [x, y, theta]
 */
void ParticleFilter::init(double x, double y, double theta, double std[]) {
#ifdef IS_DEBUG
  cout << "particle filter init begin" << endl;
#endif
  normal_distribution<double> gauss_x(0, std[0]);
  normal_distribution<double> gauss_y(0, std[1]);
  normal_distribution<double> gauss_theta(0, std[2]);

  num_particles = 150;
  particles.reserve(num_particles); // reserve vs resize debug hell, C++ $%#&
  weights.resize(num_particles);

  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.x = x + gauss_x(rnd);
    particle.y = y + gauss_y(rnd);
    particle.theta = theta + gauss_theta(rnd);
    particle.weight = 1.0;
    particle.id = i;
    particles.push_back(particle);
  }
  is_initialized = true;
#ifdef IS_DEBUG
  cout << "particle filter init end" << endl;
#endif
}

/**
 * Predict particle position after a delta time in its coordinate system
 * @param delta_t time delta
 * @param std_pos standard deviations for [x, y, theta]
 * @param velocity particle velocity
 * @param yaw_rate particle yaw rate
 */
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
#ifdef IS_DEBUG
  cout << "particle filter prediction begin" << endl;
#endif
  normal_distribution<double> gauss_x(0, std_pos[0]);
  normal_distribution<double> gauss_y(0, std_pos[1]);
  normal_distribution<double> gauss_theta(0, std_pos[2]);

  for (Particle &particle: particles) {
    if (abs(yaw_rate) > EPSILON) {
      double scale = velocity / yaw_rate;
      double yawd = yaw_rate * delta_t;
      double xd = scale * (sin(particle.theta + yawd) - sin(particle.theta));
      double yd = scale * (cos(particle.theta) - cos(particle.theta + yawd));
      double thetad = yawd;
      particle.x += xd;
      particle.y += yd;
      particle.theta += thetad;
    } else {
      // more-less a straight line
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    }
    particle.x += gauss_x(rnd);
    particle.y += gauss_y(rnd);
    particle.theta += gauss_theta(rnd);
  }
#ifdef IS_DEBUG
  cout << "particle filter prediction end" << endl;
#endif
}

/**
 * Associates landmark observations from particles with what we predict them to be at assuming correct coordinate system
 * @param predicted location of landmarks as we expect them to be
 * @param observations particle predictions about locations of landmarks
 */
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
#ifdef IS_DEBUG
  cout << "data association begin" << endl;
#endif
  for (auto &observation: observations) {
    double nearest_distance = numeric_limits<double>::max();
    for (auto &prediction: predicted) {
      double dx = prediction.x - observation.x;
      double dy = prediction.y - observation.y;
      double distance = sqrt(dx * dx + dy * dy);
      if (distance < nearest_distance) {
        nearest_distance = distance;
        observation.id = prediction.id;
      }
    }
  }
#ifdef IS_DEBUG
  cout << "data association end" << endl;
#endif
}

/**
 * Updates particle weights depending on expected match with what we believe and what particle senses
 * @param sensor_range sensor range
 * @param std_landmark standard deviation of landmark measurement
 * @param observations observations in vehicle's coordinate system
 * @param map_landmarks
 */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
#ifdef IS_DEBUG
  cout << "particle filter update weights begin" << endl;
#endif

  for (int i = 0; i < num_particles; i++) {
    // convert observations to map's space
    auto particle = particles[i];

    // particle is in map coordinates, we need to offset observations to these coordinates
    // we know angle and (x, y) in particle's coordinate system, hence conversion is trivial
    vector<LandmarkObs> particle_observations;
    particle_observations.reserve(observations.size());
    for (auto observation: observations) {
      auto particle_observation = LandmarkObs();
      particle_observation.id = -1;
      particle_observation.x = particle.x + observation.x * cos(particle.theta) - observation.y * sin(particle.theta);
      particle_observation.y = particle.y + observation.x * sin(particle.theta) + observation.y * cos(particle.theta);
      particle_observations.push_back(particle_observation);
    }

    // get landmarks within sensor range of a particle
    vector<LandmarkObs> nearby_landmarks;
    for (auto &landmark: map_landmarks.landmark_list) {
      double dx = particle.x - landmark.x_f;
      double dy = particle.y - landmark.y_f;
      double distance = sqrt(dx * dx + dy * dy);
      if (distance <= sensor_range)
        nearby_landmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
    }

    // nearest landmarks assignment
    dataAssociation(nearby_landmarks, particle_observations);

    // update particle weight
    double weight = 1.;
    for (auto &particle_observation: particle_observations) {
      // find associated landmark
      bool has_found = false;
      double lx = 0;
      double ly = 0;
      if (particle_observation.id != -1) {
        for (auto &landmark: nearby_landmarks) {
          if (particle_observation.id == landmark.id) {
            has_found = true;
            lx = landmark.x;
            ly = landmark.y;
          }
        }
        if (!has_found) {
          // no landmark in sensor vicinity, nothing to do, particle should die off
          weights[i] = 0.;
          continue;
        }
        double dx = particle_observation.x - lx;
        double dy = particle_observation.y - ly;
        double sx = std_landmark[0];
        double sy = std_landmark[1];
        weight *= 1 / (M_2PI * sx * sy) * exp(-((dx * dx) / (2 * sx * sx) + (dy * dy) / (2 * sy * sy)));
      }
    }
    weights[i] = weight;
  }
  // normalize weights (might not be necessary due to chosen resampling algorithm)
  double scaling = 0;
  for (int i = 0; i < num_particles; i++)
    scaling += weights[i];
  for (int i = 0; i < num_particles; i++)
    weights[i] = weights[i] / scaling;
#ifdef IS_DEBUG
  cout << "particle filter update weights end" << endl;
#endif
}

/**
 * Resample particles according to their individual weights
 */
void ParticleFilter::resample() {
#ifdef IS_DEBUG
  cout << "particle filter resampling begin" << endl;
#endif
  discrete_distribution<> distribution(weights.begin(), weights.end());
  vector<Particle> resampled_particles;
  resampled_particles.reserve(weights.size());
  for (int i = 0; i < num_particles; i++) {
    int j = distribution(rnd);
    resampled_particles.push_back(Particle {particles[j].id, particles[j].x, particles[j].y, particles[j].theta, particles[j].weight});
  }
  particles = resampled_particles;
#ifdef IS_DEBUG
  cout << "particle filter resampling end" << endl;
#endif
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
