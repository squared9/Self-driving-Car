#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;

#define LANE_WIDTH 4
#define HALF_LANE_WIDTH (LANE_WIDTH / 2)
#define SAFE_LANE_MARGIN 0.4
#define LOOK_AHEAD_DISTANCE 40
#define DANGEROUS_DISTANCE (LOOK_AHEAD_DISTANCE / 2)
#define SPEED_LIMIT 49.5
#define ACCELERATION  0.224
#define EMERGENCY_ACCELERATION (3 * ACCELERATION)
#define NUMBER_OF_WAYPOINTS 50
#define MS_TO_MPH 2.2369362920544
#define TIME_INTERVAL 0.02
#define NUMBER_OF_LANES 3
#define LANE_CHANGE_TIME 2.0
#define LANE_CHANGE_SAFE_ZONE 20
#define SPEED_TO_DISTANCE_RANK_RATIO 1
#define UNUSED_LANE 1E10

// sensor fusion
//[id, x, y, vx, vy, s, d]
#define SF_ID 0
#define SF_X 1
#define SF_Y 2
#define SF_VX 3
#define SF_VY 4
#define SF_S 5
#define SF_D 6

int current_lane = 1; // lane the car currently drives in
int changing_lane = -1; // lane car tries to switch to right now; -1 if no change wanted
//    double current_speed = 49.5; //mph
double current_speed = 0; // current car's speed in mph

// for convenience
bool isSafelyWithin(int lane, double car_d);

using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2( (map_y-y),(map_x-x) );

	double angle = abs(theta-heading);

	if(angle > pi()/4)
	{
		closestWaypoint++;
	}

	return closestWaypoint;

}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, vector<double> maps_s, vector<double> maps_x, vector<double> maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

vector<double> rebase(double x, double y, double angle) {
  vector<double> result;
  result.push_back(x * cos(angle) - y * sin(angle));
  result.push_back(x * sin(angle) + y * cos(angle));
  return result;
}

void getAvailableLanes(nlohmann::basic_json<map, vector, string, bool, long, unsigned long, double, std::allocator,
        nlohmann::adl_serializer> sensor_fusion, int current_lane, double current_speed, double car_s, double car_d, double available_lanes[]) {

  // double available_lanes[NUMBER_OF_LANES];

  // reset available lanes ranking
  for (int i = 0; i < NUMBER_OF_LANES; i++) {
    available_lanes[i] = UNUSED_LANE;
  }

  // predict car's s position after lane change given constant speed
  double pred_car_s = car_s + LANE_CHANGE_TIME * current_speed;

  for (int i = 0; i < sensor_fusion.size(); i++) {
    int vehicle_id = sensor_fusion[i][SF_ID];
    float traffic_s = sensor_fusion[i][SF_S];
    float traffic_d = sensor_fusion[i][SF_D];
    double traffic_vx = sensor_fusion[i][SF_VX];
    double traffic_vy = sensor_fusion[i][SF_VY];
    double traffic_speed = sqrt(traffic_vx * traffic_vx + traffic_vy * traffic_vy);
    double traffic_lane_position = traffic_d / LANE_WIDTH;
    int traffic_lane = floor(traffic_lane_position);

    cout << "-------" << endl;
    cout << "traffic id: " << vehicle_id << " s=" << traffic_s << " d=" << traffic_d << " vx=" << traffic_vx
         << " vy=" << traffic_vx << " lane=" << traffic_lane_position << " car_s=" << car_s << " car_d=" << car_d << endl;

    // ignore current traffic lane
    if (traffic_lane == current_lane)
      continue;

    // multiple lane changes need more time
    double lane_difference = abs(traffic_lane - current_lane);

    // if lane is already blocked for overtaking, ignore it
    if (available_lanes[traffic_lane] == std::numeric_limits<double>::max())
      continue;

    double speed_difference = current_speed - traffic_speed;
    double absolute_speed_difference = abs(speed_difference);

    double distance_difference = car_s - traffic_s;
    double absolute_distance_difference = abs(distance_difference);

    // predict traffic position after lane change given constant speed
    double pred_traffic_s = traffic_s + LANE_CHANGE_TIME * traffic_speed * lane_difference;
    double predicted_distance_difference = pred_car_s - pred_traffic_s;
    double absolute_predicted_distance_difference = abs(predicted_distance_difference);

    // ---------------------------
    // Non-collision checks first:
    // ---------------------------

    // if either current or predicted position will be blocked, block the lane for changing
    if (absolute_distance_difference <= LANE_CHANGE_SAFE_ZONE ||
        absolute_predicted_distance_difference <= LANE_CHANGE_SAFE_ZONE) {
      available_lanes[traffic_lane] = std::numeric_limits<double>::max();
      continue;
    }

    // if car in lane is too fast that it speeds past the car during lane change maneuver or too slow
    if (predicted_distance_difference > 0 && distance_difference < 0 ||  // car too fast
        predicted_distance_difference < 0 && distance_difference > 0) {  // car too slow
      available_lanes[traffic_lane] = std::numeric_limits<double>::max();
      continue;
    }

    // -----------------
    // Ranking of lanes:
    // -----------------

    // car doesn't seem to be blocking, now rank each car depending on distance/speed to choose the best lane
    // the best car is the one that is faster and farthest ahead or slower and farthest behind

    double rank = 0;

    rank = distance_difference - SPEED_TO_DISTANCE_RANK_RATIO * speed_difference;
    if (distance_difference < 0) {
      rank = -rank;
    }

    // take minimal rank for each lane
    if (available_lanes[traffic_lane] > rank)
      available_lanes[traffic_lane] = rank;
  }
  cout << "Lane ranking: ";
  for (int i = 0; i < NUMBER_OF_LANES; i++) {
    cout << i << ": " << available_lanes[i];
  }
  cout << endl;
}

/**
 *
 * @param available_lanes
 * @param currentLane
 * @param wantedLane
 * @return
 */
bool checkLaneTransition(double available_lanes[], int currentLane, int wantedLane) {
  bool result = true;
  int start = currentLane;
  int end = wantedLane;
  if (start > end) {
    start = wantedLane;
    end = currentLane;
  }
  for (int i = start + 1; i < end; i++) {
    // blocked lane is -1
    if (available_lanes[i] >= UNUSED_LANE)
      return false;
  }
  return result;
}

/**
 * Picks a best-ranked available lane
 * @param available_lanes ranked lanes
 * @param current_lane current car's lane
 * @return
 */
int pickLane(double available_lanes[], int current_lane) {
  int result = -1;
  double max = -1;
  for (int i = 0; i < NUMBER_OF_LANES; i++) {
    // skip blocked lanes
    if (i == current_lane || available_lanes[i] >= UNUSED_LANE)
      continue;
    // allow only transitions not blocked by other lanes
    if (!checkLaneTransition(available_lanes, current_lane, i))
      continue;
    // find highest ranked lane
    if (available_lanes[i] > max) {
      max = available_lanes[i];
      result = i;
    }
  }
  return result;
}

/**
 *
 * @param lane
 * @param car_d
 * @return
 */
bool isSafelyWithin(int lane, double car_d) {
  return car_d > lane * LANE_WIDTH + SAFE_LANE_MARGIN && car_d < (lane + 1) * LANE_WIDTH - SAFE_LANE_MARGIN ;
}

/**
 * Returns distance to nearest car ahead in the same lane
 * @param sensor_fusion current sensor fusion data
 * @param current_lane current car's lane
 * @param car_s car's Frenet s coordinate
 * @returns distance to nearest car ahead in the same lane
 */
double getAheadDistance(nlohmann::basic_json<map, vector, string, bool, long, unsigned long, double, std::allocator,
    nlohmann::adl_serializer> sensor_fusion, int current_lane, double car_s) {

  double min_distance = std::numeric_limits<double>::max();

  for (int i = 0; i < sensor_fusion.size(); i++) {
    int vehicle_id = sensor_fusion[i][SF_ID];
    float traffic_s = sensor_fusion[i][SF_S];
    float traffic_d = sensor_fusion[i][SF_D];
    double traffic_vx = sensor_fusion[i][SF_VX];
    double traffic_vy = sensor_fusion[i][SF_VY];
    double traffic_speed = sqrt(traffic_vx * traffic_vx + traffic_vy * traffic_vy);
    double traffic_lane_position = traffic_d / LANE_WIDTH;
    int traffic_lane = floor(traffic_lane_position);

    // ignore current traffic lane
    if (traffic_lane != current_lane)
      continue;

    double distance = traffic_s - car_s;

    if (distance > 0 && distance < min_distance)
      min_distance = distance;
  }
  return min_distance;
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy/* [uncomment for gcc], &current_lane, &current_speed, &changing_lane*/](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;

    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
        	// Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

          json msgJson;

          int prev_size = previous_path_x.size();

          if (prev_size > 0) {
            car_s = end_path_s;
          }

          bool too_close = false;

          for (int i = 0; i < sensor_fusion.size(); i++) {
            float d = sensor_fusion[i][SF_D];
            if ((d > LANE_WIDTH * current_lane) && (d < LANE_WIDTH * (current_lane + 1))) {
              double vx = sensor_fusion[i][3];
              double vy = sensor_fusion[i][4];
              double check_speed = sqrt(vx * vx + vy * vy);
              double check_car_s = sensor_fusion[i][SF_S];

              check_car_s += ((double) prev_size) * TIME_INTERVAL * check_speed;

              if (check_car_s > car_s && (check_car_s - car_s) < LOOK_AHEAD_DISTANCE) {
                too_close = true;
              }
            }
          }

          if (too_close) {
            if (changing_lane == -1) {
              double available_lanes[NUMBER_OF_LANES];
              getAvailableLanes(sensor_fusion, current_lane, current_speed, car_s, car_d, available_lanes);
              int laneToGo = pickLane(available_lanes, current_lane);
              if (laneToGo != -1) {
                current_lane = laneToGo;
                changing_lane = laneToGo;
              }
              else {
                double ahead_distance = getAheadDistance(sensor_fusion, current_lane, car_s);
                if (ahead_distance < DANGEROUS_DISTANCE)
                  current_speed -= EMERGENCY_ACCELERATION;
                else
                  current_speed -= ACCELERATION;
              }
            } else {
              if (isSafelyWithin(changing_lane, car_d)) {
                changing_lane = -1;
              }
            }
          } else if (current_speed < SPEED_LIMIT ){
            current_speed += ACCELERATION;
          }

          vector<double> next_x_vals;
          vector<double> next_y_vals;

          vector<double> ptsx;
          vector<double> ptsy;

          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);

          if (prev_size > 1) {
            ref_x = previous_path_x[prev_size - 1];
            ref_y = previous_path_y[prev_size - 1];
            
            double ref_prev_x = previous_path_x[prev_size - 2];
            double ref_prev_y = previous_path_y[prev_size - 2];
            ref_yaw = atan2(ref_y - ref_prev_y, ref_x - ref_prev_x);
            
            ptsx.push_back(ref_prev_x);
            ptsy.push_back(ref_prev_y);
            
            ptsx.push_back(ref_x);
            ptsy.push_back(ref_y);
          } 
          else 
          {
            double prev_car_x = car_x - cos(car_yaw);
            double prev_car_y = car_y - sin(car_yaw);

            ptsx.push_back(prev_car_x);
            ptsy.push_back(prev_car_y);
            
            ptsx.push_back(car_x);
            ptsy.push_back(car_y);
          }

          vector<double> next_wp_0 = getXY(car_s + LOOK_AHEAD_DISTANCE, HALF_LANE_WIDTH + LANE_WIDTH * current_lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp_1 = getXY(car_s + 2 * LOOK_AHEAD_DISTANCE, HALF_LANE_WIDTH + LANE_WIDTH * current_lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp_2 = getXY(car_s + 3 * LOOK_AHEAD_DISTANCE, HALF_LANE_WIDTH + LANE_WIDTH * current_lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          
          ptsx.push_back(next_wp_0[0]);
          ptsy.push_back(next_wp_0[1]);

          ptsx.push_back(next_wp_1[0]);
          ptsy.push_back(next_wp_1[1]);

          ptsx.push_back(next_wp_2[0]);
          ptsy.push_back(next_wp_2[1]);
          
          for (int i = 0; i < ptsx.size(); i++) {
            double shift_x = ptsx[i] - ref_x;
            double shift_y = ptsy[i] - ref_y;
            
            const vector<double> &coords = rebase(shift_x, shift_y, -ref_yaw);
            ptsx[i] = coords[0];
            ptsy[i] = coords[1];
          }

          tk::spline s;
          s.set_points(ptsx, ptsy);

          for (int i = 0; i < prev_size; i++) {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }
          
          double target_x = LOOK_AHEAD_DISTANCE;
          double target_y = s(target_x);
          double target_dist = sqrt(target_x * target_x + target_y * target_y);
          
          double x_add_on = 0;
          
          for (int i = 1; i <= NUMBER_OF_WAYPOINTS - prev_size; i++) {
            double N = target_dist / (TIME_INTERVAL * current_speed / MS_TO_MPH);
            double x_point = x_add_on + target_x / N;
            double y_point = s(x_point);
            
            x_add_on = x_point;
            
            double x_ref = x_point;
            double y_ref = y_point;

            vector<double> coords = rebase(x_ref, y_ref, ref_yaw);
            x_point = coords[0];
            y_point = coords[1];

            x_point += ref_x;
            y_point += ref_y;

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          //this_thread::sleep_for(chrono::milliseconds(1000));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
