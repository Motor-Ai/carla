# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""

import carla
import torch
import matplotlib.pyplot as plt
import carla.libcarla
from shapely.geometry import Polygon
import math
import matplotlib
matplotlib.use('TKAgg', force=True)
import numpy as np
import  agents.navigation.frenet.cubic_spline_planner as cp
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.frenet.frenet_optimal_trajectory import FrenetPlanner
from agents.navigation.frenet.frenet_optimal_trajectory import closest_wp_idx
from agents.navigation.frenet.cubic_spline_planner import calc_bspline_course_2
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.frenet.frenet_mai import get_frenet_traj, global_to_egocentric, egocentric_to_global
from agents.navigation.mai_utils.vector_BEV_observer import Vector_BEV_observer

# from agents.navigation.controller import VehiclePIDController+
from agents.navigation.mpc.mpc_node import mpcControlNode
from agents.tools.misc import (get_speed, is_within_distance,
                               get_trafficlight_trigger_location,
                               compute_distance)
from agents.tools.misc import draw_waypoints, get_speed

class BasicAgent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(self, vehicle, target_speed=20, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param target_speed: speed (in Km/h) at which the vehicle will move
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.

        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()
        self._last_traffic_light = None

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._use_bbs_detection = False
        self._target_speed = target_speed
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 0.5  # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._speed_ratio = 1
        self._max_brake = 0.5
        self._offset = 0
        self.f_idx = 0
        self.vehicle_controller = mpcControlNode()
        self.bev_observer = Vector_BEV_observer()
        # Change parameters according to the dictionary
        opt_dict['target_speed'] = target_speed
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'use_bbs_detection' in opt_dict:
            self._use_bbs_detection = opt_dict['use_bbs_detection']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'detection_speed_ratio' in opt_dict:
            self._speed_ratio = opt_dict['detection_speed_ratio']
        if 'max_brake' in opt_dict:
            self._max_brake = opt_dict['max_brake']
        if 'offset' in opt_dict:
            self._offset = opt_dict['offset']

        # Initialize the planners
        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict, map_inst=self._map)

        if grp_inst:
            if isinstance(grp_inst, GlobalRoutePlanner):
                self._global_planner = grp_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)
        else:
            self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

        # Get the static elements of the scene
        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}  # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

            :param value (bool): whether or not to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

        wp_x, wp_y, wp_yaw,manuvers = [], [], [], []
        rwp_x, rwp_y, rwp_yaw = [], [], []
        lwp_x, lwp_y, lwp_yaw = [], [], []
        
        for wp,man in self._local_planner._waypoints_queue:
            # Main Path
            wp_x.append(wp.transform.location.x)
            wp_y.append(wp.transform.location.y)
            wp_yaw.append(np.deg2rad(wp.transform.rotation.yaw))
            manuvers.append(man)

            # Neighboring lanes
            right_wp = wp.get_right_lane()
            if right_wp is not None:
                rwp_x.append(right_wp.transform.location.x)
                rwp_y.append(right_wp.transform.location.y)
                rwp_yaw.append(np.deg2rad(right_wp.transform.rotation.yaw))
            else:
                rwp_x.append(wp.transform.location.x)
                rwp_y.append(wp.transform.location.y)
                rwp_yaw.append(np.deg2rad(wp.transform.rotation.yaw))

            left_wp = wp.get_left_lane()
            if left_wp is not None:
                lwp_x.append(left_wp.transform.location.x)
                lwp_y.append(left_wp.transform.location.y)
                lwp_yaw.append(np.deg2rad(left_wp.transform.rotation.yaw))
            else:
                lwp_x.append(wp.transform.location.x)
                lwp_y.append(wp.transform.location.y)
                lwp_yaw.append(np.deg2rad(wp.transform.rotation.yaw))
            
        current_lane =torch.tensor([wp_x, wp_y, wp_yaw]).T
        right_lane =torch.tensor([rwp_x, rwp_y, rwp_yaw]).T
        left_lane =torch.tensor([lwp_x, lwp_y, lwp_yaw]).T

        map_lanes = torch.stack([current_lane, right_lane, left_lane])
        # diff = torch.diff(map_lanes, dim=-2)
        # # Torch Unique doesnt work
        # non_zero_elem = torch.any(diff[...,:2,:] != 0.0)
        # non_zero_elem = non_zero_elem.expand_as(diff)
        # map_lane_unq = torch.masked_select(map_lanes[:,1:,:], non_zero_elem)
        # self.map_lanes  = torch.cat([map_lanes[:,0,:], map_lane_unq], dim=-2)
        map_lanes = torch.unique_consecutive(map_lanes, dim= -2)
        path_len = 100
        path_res = 0.5
        self.map_lanes= torch.zeros((len(map_lanes),int(path_len//path_res),3))
        for i,lane in enumerate(map_lanes):
            tx, ty, tyaw, _ = cp.calc_bspline_course_2(
                    lane[:,0].tolist(),
                    lane[:,1].tolist(),
                    path_len,
                    path_res,
            )
            self.map_lanes[i,:,0] = torch.tensor(tx)
            self.map_lanes[i,:,1] = torch.tensor(ty)
            self.map_lanes[i,:,2] = torch.tensor(tyaw)


    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self):
        """Execute one step of navigation."""
        hazard_detected = False

        # Retrieve all relevant actors
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        vehicle_speed = get_speed(self._vehicle)

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(self._lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True

        # Create state for frenet planner
        acc_vec = self._local_planner._vehicle.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        temp = [self._local_planner._vehicle.get_velocity(), self._local_planner._vehicle.get_acceleration()]
        speed = vehicle_speed
        psi = math.radians(self._local_planner._vehicle.get_transform().rotation.yaw)
        self.max_s = 20
        set_spd = 20
        ego_state = [self._local_planner._vehicle.get_location().x, self._local_planner._vehicle.get_location().y,psi, 0, 0, temp, self.max_s]

        # Transform lanes to EGO frame
        map_lanes_ego = torch.zeros_like(self.map_lanes)
        map_lanes_ego[0:1,:,:] = global_to_egocentric(torch.tensor([ego_state[0], ego_state[1], ego_state[2]]), self.map_lanes[0:1,:,:], velocity=False)
        map_lanes_ego[1:2,:,:] = global_to_egocentric(torch.tensor([ego_state[0], ego_state[1], ego_state[2]]), self.map_lanes[1:2,:,:], velocity=False)
        map_lanes_ego[2:,:,:] = global_to_egocentric(torch.tensor([ego_state[0], ego_state[1], ego_state[2]]), self.map_lanes[2:,:,:], velocity=False)
        
        map_lanes_ego = map_lanes_ego.unsqueeze(0).unsqueeze(0)

        # Consider only the path from the ego vehicle
        path = np.abs(
            np.array(
                [
                    map_lanes_ego[0, 0, 0, :, 0].cpu().detach().numpy(),
                    map_lanes_ego[0, 0, 0, :, 1].cpu().detach().numpy(),
                ]
            ).T
        )

        idx = np.argmin(np.hypot(path[:, 0], path[:, 1]))
        map_lanes_ego = map_lanes_ego[:,:,:, idx:]

        # Continue only if the length of the remaining path is greater than 1
        if map_lanes_ego.shape[-2] < 2:
            # Clear the waypoint queue, so that the controller kknows the goal is reached
            self._local_planner._waypoints_queue.clear()
            self.bev_observer.save_file(re_reference=False, file_name="test")
            self.done()
            control = carla.VehicleControl()
            control.brake = self._local_planner._max_brake
            return control

        # Get frenet trajectories
        fpath_idx, fplist = get_frenet_traj(map_lanes_ego, [0,0,0, temp[0].x, temp[0].y])

        if self.bev_observer.client_init(world=self._world):
            self.bev_observer.update_traj(fplist, plot = False, re_reference = False)
            self.bev_observer.update(plot=False, re_reference=False)

        # Transform trajectories to global frame
        for fp_ego in fplist:
            path =fp_ego[:, :3]
            data = egocentric_to_global(torch.tensor([ego_state[0], ego_state[1], ego_state[2]]), path, velocity=False)
            fp_ego[:,0] = data[:,0]
            fp_ego[:,1] = data[:,1]
            fp_ego[:,2] = data[:,2]

        best_idx = 6 #TODO: Choose the best intention point
        # try:
        cmdWP2 = [fplist[fpath_idx][best_idx,0], fplist[fpath_idx][best_idx,1]]
        # except IndexError:
        #     self.done()
        #     control = carla.VehicleControl()
        #     control.brake = self._local_planner._max_brake
        #     return control
        
        # Get frenet location in waypoint datastructure
        loc = carla.libcarla.Location()
        loc.x = cmdWP2[0].item()
        loc.y = cmdWP2[1].item()
        frenet_loc = self._local_planner._map.get_waypoint(loc)

        # Plotting
        plt.cla()
        leg = []
        plt.scatter(self.map_lanes[...,0,:,1], self.map_lanes[...,0,:,0])
        leg.extend(["Global path"])

        plt.scatter(self.map_lanes[...,1,:,1], self.map_lanes[...,1,:,0])
        plt.scatter(self.map_lanes[...,2,:,1], self.map_lanes[...,2,:,0])
        for i, p in enumerate(fplist):
            plt.plot(p[:,1], p[:,0])#, s=0.2)
            leg.append([f"candidate:{i}"])
        # # Add index as text to each point in scatter plot
        # for i, (x, y) in enumerate(zip(self.xx, self.yy)):
        #     plt.text(y,x, str(i), fontsize=8)
        plt.scatter(fplist[fpath_idx][:,1], fplist[fpath_idx][:,0], s=0.2)
        leg.extend(["Main path"])
        plt.scatter(ego_state[1],ego_state[0], marker="x")
        # if ego_state != None:
        #     # Plot a triangle with position and orientation
        #     plt.quiver(ego_state[1], ego_state[0], math.sin(ego_state[2]), math.cos(ego_state[2]), color="black", scale=20, width=0.01)
        plt.scatter(cmdWP2[1], cmdWP2[0], marker="o", color="black")
        # plt.xlim([-30 + ego_state[1], 30 + ego_state[1]])
        # plt.ylim([-30 + ego_state[0], 30 + ego_state[0]])
        plt.legend(leg)
        plt.pause(0.01)
        control = self._local_planner._vehicle_controller.run_step(20, frenet_loc)

        # # MPC
        # ego_ctrl= []
        # ego_control = self._local_planner._vehicle.get_control()
        # ego_ctrl.extend([ego_control.brake, ego_control.steer, ego_control.throttle])
        # control = self.vehicle_controller.mpc_main(ego_state, fplist[fpath_idx], ego_ctrl)

        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def set_offset(self, offset):
        """Sets an offset for the vehicle"""
        self._local_planner.set_offset(offset)

    def lane_change(self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=2):
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver
        """
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution
        )
        if not path:
            print("WARNING: Ignoring the lane change as no path was found")

        self.set_global_plan(path)

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if trigger_wp.transform.location.distance(ego_vehicle_location) > max_distance:
                continue

            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)

    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self._vehicle.bounding_box.extent.x * ego_transform.get_forward_vector())

        opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))

        return (False, None, -1)

    def _generate_lane_change_path(self, waypoint, direction='left', distance_same_lane=10,
                                distance_other_lane=25, lane_change_distance=25,
                                check=True, lane_changes=1, step_distance=2):
        """
        This methods generates a path that results in a lane change.
        Use the different distances to fine-tune the maneuver.
        If the lane change is impossible, the returned path will be empty.
        """
        distance_same_lane = max(distance_same_lane, 0.1)
        distance_other_lane = max(distance_other_lane, 0.1)
        lane_change_distance = max(lane_change_distance, 0.1)

        plan = []
        plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

        option = RoadOption.LANEFOLLOW

        # Same lane
        distance = 0
        while distance < distance_same_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        if direction == 'left':
            option = RoadOption.CHANGELANELEFT
        elif direction == 'right':
            option = RoadOption.CHANGELANERIGHT
        else:
            # ERROR, input value for change must be 'left' or 'right'
            return []

        lane_changes_done = 0
        lane_change_distance = lane_change_distance / lane_changes

        # Lane change
        while lane_changes_done < lane_changes:

            # Move forward
            next_wps = plan[-1][0].next(lane_change_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]

            # Get the side lane
            if direction == 'left':
                if check and str(next_wp.lane_change) not in ['Left', 'Both']:
                    return []
                side_wp = next_wp.get_left_lane()
            else:
                if check and str(next_wp.lane_change) not in ['Right', 'Both']:
                    return []
                side_wp = next_wp.get_right_lane()

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return []

            # Update the plan
            plan.append((side_wp, option))
            lane_changes_done += 1

        # Other lane
        distance = 0
        while distance < distance_other_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan
