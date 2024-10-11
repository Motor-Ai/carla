import time
import traceback2 as traceback
from numpy import clip, radians, degrees, hypot, pi, uint32
import matplotlib.pyplot as plt
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSPresetProfiles

import matplotlib.pyplot as plt
from pathlib import Path

## import helper files
from agents.navigation.mpc import pid_speed_control_helper as pid_speed
from agents.navigation.mpc import mpc_helper
from agents.navigation.mpc import ego_params as ep
from agents.navigation.mpc import command_converter_EQV as cmd_conv
from agents.navigation.mpc import butter_low_pass_filter as tfil

# ## import custom message
# from stack_interfaces.msg import FrenetPathMsg
# from stack_interfaces.msg import CarControl
# from stack_interfaces.msg import Canbus
# from stack_interfaces.msg import GnssOdom


NX = 5                  # [x, y, v, yaw, delta]
NY = 6
NYN = 4
NU = 2                  # [accel, deltarate]
T = 40                   # horizon length
MAX_STEER = np.deg2rad(30.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(90.0/4.5)  # maximum steering speed [rad/s]

#############################################


SIMULATION = False

SHOW_ANIMATION = False


#############################################


class mpcControlNode(Node):
    def __init__(self):
        # super().__init__("mpc_control_node")
        try:
            """
            Initialize all Class Variables
            """

            ## Frame ID
            self.i = 0

            ## MPC variables
            self.oa = [0.0] * T
            self.odelta = [0.0] * T

            ## Frenet Path Variables
            self.new_path_x, self.new_path_y, self.new_path_yaw = [], [], []
            self.new_path_k, self.new_path_speed_profile = [], []
            self.new_path_resolution = 0.1
            self.new_path_length = 0.0

            self.path_x, self.path_y = [], []
            self.path_yaw, self.path_k = [], []
            self.path_speed_profile = []
            self.path_resolution = 0.1
            self.path_length = 0.0
            self.scenario = 'LANEKEEP'
            self.lane_morphology = 'normal'
            self.speed_target_mod = 0

            ## Path Control Variables
            self.target_ind = 0

            ## Ego Variables (Ego Pose information)
            #self.ego_x, self.ego_y, self.ego_yaw, self.ego_vel = None, None, None, None
            self.ego_x, self.ego_y, self.ego_yaw, self.ego_vel = 0.0, 0.0, 0.0, 0.0
            self.state = None
            self.localization_checksum = 0
            self.ego_steer_val = 0.0
            self.ego_brake_val = 0.0
            self.ego_throttle_val = 0.0
            self.ego_throttle_val_rate = 0.0
            self.lat_dist = 0.0
            self.canbus_checksum = False
            e_params = ep.parameters_EQV()
            self.steer_ratio = e_params.steering.max / e_params.steering.wheel_max
            self.wheel_base = e_params.a + e_params.b  # Wheel Base of Ego Vehicle
            self.steer_max = e_params.steering.max

            ## Commands
            self.last_ai = 0.0  # last acceleration
            self.last_di = 0.0
            self.dbw_reaction_delay = 0.3
            self.throttle_cmd_based_aux_delay = 0.0
            self.brake_cmd_based_aux_delay = 0.0
            self.total_delay = 0.0
            self.front_axle_xy = [0.0,0.0]


            self.steering_cmd_arr = [0.0] * 30
            self.steer_filter_window = 150
            self.last_throttle = 0.0
            self.throttle_gas_level = 0.0
            self.standby_max_brake = -0.25
            self.standby_max_steer_rate = 200.0
            self.steer_max_error = 400.0

            self.x_pred, self.y_pred = [], []
            self.x_ref, self.y_ref = [], []
            self.yaw_ref = []
            self.front_axle_idx = 0
            self.error_front_axle = 0.0
            self.weight = 0.0

            ## Safety Flags
            self.driver_brake = False
            if SIMULATION == True:
                self.vehicle_rolling = True
            else:
                self.vehicle_rolling = False

            ## PID Parameters
            self.I_term_prev = 0.0

            ## Navigation
            self.direction_of_travel = "F"

            # Time per frame
            self.DT = 1 / 20.0  # [s] time tick

            # self.frenet_header = self.get_clock().now().to_msg()
            self.time_delay_gridmap_2_control = 0.0

            ## Health Monitoring Variables
            self.frenet_timer = 0.0
            self.localization_timer = 0.0
            self.frenet_first_msg = False
            self.localization_first_msg = False
            self.frenet_comms_lost = False
            self.localization_comms_lost = False
            self.frenet_timeout = 3.0
            self.localization_timeout = 3.0

            self.init_params()
            
            # self.subscription_3 = self.create_subscription(
            #     FrenetPathMsg, "/frenet_path", self.path_callback, QoSPresetProfiles.SYSTEM_DEFAULT.value
            # )
            # self.canbus_sub = self.create_subscription(
            #     Canbus, "/shadow/canbus", self.canbus_callback, QoSPresetProfiles.SENSOR_DATA.value
            # )
            # self.canbus_sub = self.create_subscription(
            #     Canbus, "/shadow/canbus", self.canbus_callback, QoSPresetProfiles.SENSOR_DATA.value
            # )
            # subscribe to localization topic

            # if SIMULATION:
            #     self.subscription_1 = self.create_subscription(
            #         GnssOdom,
            #         "/localization/filtered_output/sim",
            #         self.egopose_callback,
            #         QoSPresetProfiles.SENSOR_DATA.value,
            #     )
            # else:
            #     self.subscription_1 = self.create_subscription(
            #         GnssOdom,
            #         "/localization/filtered_output",
            #         self.egopose_callback,
            #         QoSPresetProfiles.SENSOR_DATA.value,
            #     )



            # self.publisher_1 = self.create_publisher(CarControl, "/ego_control_cmds", QoSPresetProfiles.SYSTEM_DEFAULT.value)

            # rclpy doesn't seem to have the clock object fully implemented as in C++. This node aims at being converted to cpp for better timing.
            # self.timer = self.create_timer(
            #     timer_period_sec=0.05, callback=self.control_callback
            # )

            # rclpy.time.Duration(seconds=0.05)
            # for later :
            # ROSclock = Clock  #self.get_clock().now()
            # self.timer = self.create_timer(clock=ROSclock, timer_period_sec=rclpy.time.Duration(seconds=0.05), callback=self.control_callback)

            ## MPC Control Options
            self.vehicle_ready = False
            self.start_vehicle()

            print("MPC Control module initialized!")

        except Exception as e:
            print(
                "MPC Control initializing error : "
                + str(e)
                + str(traceback.format_exc())
            )


    def init_params(self):
        self.Kp_v = 0.15
        self.tauI_v = 10.0
        self.max_throttle_change_accel = 0.5
        self.max_throttle_change_accel_release = 0.9
        self.max_throttle_change_brake = 0.7
        self.max_throttle_change_brake_release = 0.7
        self.DT = 0.05
        self.standby_max_brake = -0.25
        self.standby_max_steer_rate = 200.0
        self.steer_max_error = 400.0
        self.steer_filter_window = 150

        # ## Safety Flags
        # self.driver_brake = False
        # if SIMULATION == True:
        #     self.vehicle_rolling = True
        # else:
        #     self.vehicle_rolling = False
        # self.declare_parameter("Kp_v", 0.15)
        # self.declare_parameter("tauI_v", 10.0)
        # self.declare_parameter("max_throttle_change_accel", 0.5)
        # self.declare_parameter("max_throttle_change_accel_release", 0.9)
        # self.declare_parameter("max_throttle_change_brake", 0.7)
        # self.declare_parameter("max_throttle_change_brake_release", 0.7)
        # self.declare_parameter("DT", 0.05)
        # self.declare_parameter("standby_mode_max_brake", -0.25)
        # self.declare_parameter("standby_mode_max_steer_rate", 200.0)
        # self.declare_parameter("normal_mode_max_steer_error", 400.0)
        # self.declare_parameter("normal_mode_steer_filter_window", 150)

        # # get parameters from configuration file
        # self.Kp_v = (
        #     self.get_parameter("Kp_v").get_parameter_value().double_value
        # )  # PID Speed Control proportional gain
        # self.tauI_v = (
        #     self.get_parameter("tauI_v").get_parameter_value().double_value
        # )  # PID Speed Control integral gain
        # self.max_throttle_change_accel = (
        #     self.get_parameter("max_throttle_change_accel")
        #     .get_parameter_value()
        #     .double_value
        # )
        # self.max_throttle_change_accel_release = (
        #     self.get_parameter("max_throttle_change_accel_release")
        #     .get_parameter_value()
        #     .double_value
        # )
        # self.max_throttle_change_brake = (
        #     self.get_parameter("max_throttle_change_brake")
        #     .get_parameter_value()
        #     .double_value
        # )
        # self.max_throttle_change_brake_release = (
        #     self.get_parameter("max_throttle_change_brake_release")
        #     .get_parameter_value()
        #     .double_value
        # )

        # # Time per frame
        # self.DT = (
        #     self.get_parameter("DT").get_parameter_value().double_value
        # )  # [s] time tick

        # self.DT = 0.05

        # self.standby_max_brake = (
        #     self.get_parameter("standby_mode_max_brake")
        #     .get_parameter_value()
        #     .double_value
        # )

        # self.standby_max_steer_rate = (
        #     self.get_parameter("standby_mode_max_steer_rate")
        #     .get_parameter_value()
        #     .double_value
        # )

        # self.steer_max_error = (
        #     self.get_parameter("normal_mode_max_steer_error")
        #     .get_parameter_value()
        #     .double_value
        # )

        # self.steer_filter_window = (
        #     self.get_parameter("normal_mode_steer_filter_window")
        #     .get_parameter_value()
        #     .integer_value
        # )

        print("Config Parameters Imported")



    def start_vehicle(self):
        """
        Sets the Flag to Start mpc Control
        """
        self.vehicle_ready = True
        print("MPC Control has started ...")



    def stop_vehicle(self):
        """
        Sets the Flag to Start mpc Control
        """
        self.vehicle_ready = False

        print("MPC Control has stopped ...")



    def canbus_callback(self, canbus_msg):
        """
        Get Canbus readings - Steer Value & Checksum
        """
        try:
            if SIMULATION == False:
                if canbus_msg.checksum == 1:
                    self.ego_steer_val = (
                        radians(canbus_msg.steering) * self.steer_ratio * -1.0
                    )

                    self.ego_brake_val = canbus_msg.brake

                    if self.ego_throttle_val == 0.0:
                        self.ego_throttle_val_rate = 0.0
                    else:
                        self.ego_throttle_val_rate = canbus_msg.throttle - self.ego_throttle_val

                    self.ego_throttle_val = canbus_msg.throttle

                    self.canbus_checksum = True
                else:
                    self.canbus_checksum = False
            else:
                ## Update Steering Val from Previous Commands
                self.ego_steer_val = self.last_di 
                self.ego_throttle_val_rate = 0.0

                if self.last_throttle > 0.0:
                    self.ego_throttle_val = float(abs(self.last_throttle * 100.0))
                    self.ego_brake_val = 0.0
                else:
                    self.ego_throttle_val = 0.0
                    self.ego_brake_val = float(abs(self.last_throttle * 100.0))
        except Exception as e:
            print(
                "Canbus Callback Error : " + str(e) + str(traceback.format_exc())
            )


    def egopose_callback(self, locmsg):
        """
        Extracts & Decodes Ego Pose Message from Localization Module
        """
        try:
            pose_x = locmsg.filtered_odometry.pose.pose.position.x
            pose_y = locmsg.filtered_odometry.pose.pose.position.y

            q_x = locmsg.filtered_odometry.pose.pose.orientation.x
            q_y = locmsg.filtered_odometry.pose.pose.orientation.y
            q_z = locmsg.filtered_odometry.pose.pose.orientation.z
            q_w = locmsg.filtered_odometry.pose.pose.orientation.w
            roll, pitch, yaw = mpc_helper.euler_from_quaternion(q_x, q_y, q_z, q_w)

            velocity = clip(
                locmsg.filtered_odometry.twist.twist.linear.x, 0.0, 140 / 3.6
            )

            self.ego_x = pose_x
            self.ego_y = pose_y
            self.ego_yaw = yaw
            self.ego_vel = velocity
            self.localization_checksum = locmsg.checksum
            
            self.localization_timer = 0.0
            if self.localization_first_msg == False:
                self.localization_first_msg = True
        except Exception as e:
            print(
                "MPC Localization Callback Error: "
                + str(e)
                + str(traceback.format_exc())
            )


    def calculate_time_difference(self, header_stamp):
        current_stamp = self.get_clock().now()
        current_time = current_stamp.seconds_nanoseconds()[0] + current_stamp.seconds_nanoseconds()[1] * 1e-9
        header_time = header_stamp.sec + header_stamp.nanosec * 1e-9
        self.time_delay_gridmap_2_control = current_time - header_time
        print(f"*************PerceptionGridmap to Control DELAY*****************  {self.time_delay_gridmap_2_control}")


    def path_callback(self, pathmsg):
        """
        Read the path message & update the respective variables
        """
        try:
            if self.ego_x and len(pathmsg.path_x) > 0:
                goal_dist = hypot(
                    pathmsg.path_x[-1] - self.ego_x, pathmsg.path_y[-1] - self.ego_y
                )
            else:
                goal_dist = 1000.0

            self.frenet_header = pathmsg.header.stamp
            self.calculate_time_difference(self.frenet_header)

            self.new_path_x = pathmsg.path_x
            self.new_path_y = pathmsg.path_y
            self.new_path_yaw = pathmsg.path_yaw
            self.new_path_k = pathmsg.path_k
            self.new_path_speed_profile = pathmsg.path_speed_profile
            self.new_path_resolution = pathmsg.path_resolution
            self.new_path_length = len(self.new_path_x)*self.new_path_resolution
            self.scenario = pathmsg.scenario
            self.lane_morphology = pathmsg.lane_section

            self.frenet_timer = 0.0
            if self.frenet_first_msg == False:
                self.frenet_first_msg = True

        except Exception as e:
            print(
                "MPC Path Callback Error : " + str(e) + str(traceback.format_exc())
            )


    # def control_callback(self):
    #     """
    #     Runs mpc Control Loop & Publishes the Control Commands to be sent to the Vehicle
    #     """
    #     try:
    #         if not (SIMULATION):
    #             if self.vehicle_rolling == False:
    #                 if self.ego_vel:
    #                     if self.ego_vel >= 4.0 / 3.6:
    #                         self.vehicle_rolling = True
    #             else:
    #                 if self.ego_vel:
    #                     if self.ego_vel <= 1.0 / 3.6:
    #                         self.vehicle_rolling = False

    #         self.mpc_main()

    #         egocontrolmsg = CarControl()
    #         #egocontrolmsg.header.stamp = self.get_clock().now().to_msg()
    #         egocontrolmsg.header.stamp = self.frenet_header
    #         egocontrolmsg.header.frame_id = "base_link"

    #         ## Write to msg
    #         egocontrolmsg.steering = -float(degrees(self.last_di) / self.steer_ratio)

    #         if self.last_throttle > 0.0:
    #             egocontrolmsg.throttle = float(abs(self.last_throttle * 100.0))
    #             egocontrolmsg.brake = 0.0
    #         else:
    #             egocontrolmsg.throttle = 0.0

    #             if self.last_throttle < -0.4:
    #                 self.last_throttle = -0.4
    #             egocontrolmsg.brake = float(abs(self.last_throttle * 100.0))

    #         egocontrolmsg.last_di = self.last_di
    #         egocontrolmsg.throttle_gas_level = self.throttle_gas_level
    #         egocontrolmsg.x_pred = self.x_pred
    #         egocontrolmsg.y_pred = self.y_pred
    #         egocontrolmsg.target_ind = int(self.target_ind)

    #         egocontrolmsg.crosstrack_dx_err = self.x_ref
    #         egocontrolmsg.crosstrack_dy_err = self.y_ref
    #         egocontrolmsg.yaw_pred = self.yaw_ref
    #         egocontrolmsg.combined_err = self.error_front_axle
    #         egocontrolmsg.p_term = self.speed_target_mod*3.6
    #         egocontrolmsg.i_term = self.ego_vel*3.6
    #         egocontrolmsg.k = self.weight
    #         egocontrolmsg.perception_2_control_delay = self.time_delay_gridmap_2_control

    #         # Publish the message to the topic
    #         self.publisher_1.publish(egocontrolmsg)
    #         self.i += 1
    #     except Exception as e:
    #         print(
    #             "MPC Control Callback Error: " + str(e) + str(traceback.format_exc())
    #         )

    def comms_monitoring(self):
        self.frenet_timer += self.DT
        self.localization_timer += self.DT

        if self.frenet_first_msg:
            if self.frenet_timer > self.frenet_timeout:
                if not (self.frenet_comms_lost):
                    self.frenet_comms_lost = True

        if self.localization_first_msg:
            if self.localization_timer > self.localization_timeout:
                if not (self.localization_comms_lost):
                    self.localization_comms_lost = True


    def plot_mpc(self, x_pred, y_pred):
        displayed_map_size_in_m = 40.0
        plt.cla()
        plt.title("SIMULATION " + str(SIMULATION)
            + "\nmpc Control:   Ego_Vel: " + str(round(self.ego_vel*3.6,1)) + " kph   Weight: " + str(self.weight) + "  Target_Vel: " + str(round(self.speed_target_mod*3.6,1)) + " kph  " + str(self.lane_morphology) + " section   Reaction_Delay: " 
            + str(round(self.total_delay,2)) + " secs", fontsize = 11)
        plt.scatter(self.ego_x, self.ego_y, c='r', label='Ego_Position')
        plt.plot(self.path_x, self.path_y, '-.m', label='Frenet_Path')
        plt.plot(x_pred, y_pred, marker ='+')
        mpc_helper.plot_arrow(self.ego_x, self.ego_y, self.ego_yaw)
        car_outline = mpc_helper.get_car_outline(self.ego_x, self.ego_y, self.ego_yaw).T
        plt.plot(car_outline[0], car_outline[1], c="k", lw=2.0)

        #plt.scatter(self.state.x, self.state.y, c='g', label='Ego_Position_2')
        plt.scatter(self.front_axle_xy[0], self.front_axle_xy[1], c='b', label='Delay_incorporated_position')
        if len(x_pred):
            plt.scatter(x_pred[self.front_axle_idx], y_pred[self.front_axle_idx], c='m', label='Error Point')
        # plt.scatter(x_pred[self.front_axle_idx], y_pred[self.front_axle_idx], c='m', label='Error Point')

        mpc_helper.plot_arrow(self.state.x, self.state.y, self.state.yaw)
        car_outline = mpc_helper.get_car_outline(self.state.x, self.state.y, self.state.yaw).T
        plt.plot(car_outline[0], car_outline[1], c="k", linestyle='dashed', lw=2.0)

        plt.scatter(10000.0,10000.0,label="Steering_Cmd: " + str(round(-float(degrees(self.last_di) / self.steer_ratio),1)) + '°', c='white')
        plt.scatter(10000.0,10000.0,label="Steering_Val: " + str(round(-float(degrees(self.ego_steer_val) / self.steer_ratio),1)) + '°', c='white')
        plt.scatter(10000.0,10000.0,label="Throttle_Val: " + str(round(self.ego_throttle_val,1)) + ' %', c='white')
        plt.scatter(10000.0,10000.0,label="Brake_Val: " + str(round(self.ego_brake_val,1)) + ' %', c='white')
        #plt.scatter(10001.0,10000.0,label="Throttle_Cmd: " + str(round(self.throttle,1)) + '%', c='white')
        #plt.scatter(10002.0,10000.0,label="Brake_Cmd: " + str(round(self.brake,1)) + '%', c='white')

        plt.xlim(
            self.ego_x - (displayed_map_size_in_m / 2),
            self.ego_x + (displayed_map_size_in_m / 2),
        )
        plt.ylim(
            self.ego_y - (displayed_map_size_in_m / 2),
            self.ego_y + (displayed_map_size_in_m / 2),
        )
        plt.legend()
        plt.grid()
        plt.pause(1)


    def mpc_main(self, ego_state, path, ego_control):

        """
        cx: course x position list
        cy: course y position list
        cy: course yaw position list
        ck: course curvature list
        sp: speed profile
        dl: course tick [m]
        """
        # Extract the path from the received path
        self.vehicle_ready = True
        self.localization_checksum = True

        self.frenet_comms_lost = False
        self.localization_comms_lost = False
        self.ego_x, self.ego_y, self.ego_yaw = ego_state[0], ego_state[1], ego_state[2]
        self.ego_vel = ego_state[3]
        self.new_path_x, self.new_path_y, self.new_path_speed_profile = path[0], path[1], path[3]
        self.new_path_yaw = mpc_helper.smooth_yaw(path[2])

        self.new_path_resolution  = 0.1
        self.new_path_length = len(self.new_path_x)*self.new_path_resolution
        
        self.ego_brake_val = ego_control[0]
        self.ego_steer_val = ego_control[1]
        self.ego_throttle_val = ego_control[2]

        if self.ego_throttle_val == 0.0:
            self.ego_throttle_val_rate = 0.0
        else:
            self.ego_throttle_val_rate = self.last_throttle - self.ego_throttle_val

        try:
            # self.comms_monitoring()

            if self.frenet_comms_lost:
                print(
                    "Frenet Communication Lost: "
                    + "  Time since last Frenet Path msg: "
                    + str(self.frenet_timer)
                    + " secs"
                )

            if self.localization_comms_lost:
                print(
                    "Localization Communication Lost: "
                    + "  Time since last Localization msg: "
                    + str(self.localization_timer)
                    + " secs"
                )

            if (
                len(self.new_path_x) > 0
                and len(self.new_path_y) > 0
                and self.ego_x != None
                and self.localization_checksum
                and self.vehicle_ready
                and not (self.frenet_comms_lost)
                and not (self.localization_comms_lost)
            ):
                ## Update Path & States when new path is received
                if self.new_path_x != self.path_x or self.new_path_y != self.path_y or self.new_path_speed_profile != self.path_speed_profile:
                    self.path_x = self.new_path_x
                    self.path_y = self.new_path_y
                    self.path_yaw = self.new_path_yaw
                    #self.path_yaw = mpc_helper.smooth_yaw(self.new_path_yaw)
                    self.path_speed_profile = self.new_path_speed_profile
                    self.path_resolution = self.new_path_resolution
                    self.path_length = self.new_path_length

                    self.state = mpc_helper.State(
                        x=self.ego_x,
                        y=self.ego_y,
                        yaw=self.ego_yaw,
                        v=self.ego_vel,
                        dt=self.DT,
                        delta=self.ego_steer_val,
                    )

                    self.target_ind,_ = mpc_helper.calc_nearest_index(self.state, self.path_x, self.path_y, self.path_yaw)

                    ## initial yaw compensation
                    if self.state.yaw - self.path_yaw[self.target_ind] >= pi:
                        self.state.yaw -= pi * 2.0
                    elif self.state.yaw - self.path_yaw[self.target_ind] <= -pi:
                        self.state.yaw += pi * 2.0


                self.state = mpc_helper.State(
                    x=self.ego_x,
                    y=self.ego_y,
                    yaw=self.ego_yaw,
                    v=self.ego_vel,
                    dt=self.DT,
                    delta=self.ego_steer_val,
                )
                
                xref, self.target_ind, dref, self.lat_dist = mpc_helper.calc_ref_trajectory(self.state, self.path_x, self.path_y, self.path_yaw, self.path_k, self.path_speed_profile, 0.1)
                xref[3] = mpc_helper.smooth_yaw(xref[3])

                self.x_ref = xref[0].tolist()
                self.y_ref = xref[1].tolist()
                self.yaw_ref = xref[3].tolist()

                # self.state = mpc_helper.delay_corrected_state(self.state, 0.5, 10.0/3.6, 20.0/3.6)

                x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw, self.state.delta]  # current state

                ###### should xref and dref be zeros or taken from references.
                #xref = np.zeros((NY, T + 1))
                #dref = np.zeros((1, T + 1))
                ###### dref has always been used as zero. whereas xref can be taken from reference path.

                self.oa, self.odelta, ox, oy, oyaw, ov, self.weight = mpc_helper.iterative_linear_mpc_control(self, xref, x0, dref, self.oa, self.odelta, self.state.v, self.lane_morphology, self.speed_target_mod, self.ego_throttle_val_rate)

                if self.odelta is not None:
                    self.x_pred = ox.tolist()
                    self.y_pred = oy.tolist()

                    self.throttle_cmd_based_aux_delay = clip(self.last_throttle,0.0,1.0)* (1/6.0) 
                    self.brake_cmd_based_aux_delay = clip(self.last_throttle,-1.0,0.0)* (1/4.0) 
                    self.total_delay = self.dbw_reaction_delay + self.throttle_cmd_based_aux_delay# + self.brake_cmd_based_aux_delay
                    self.total_delay = clip(self.total_delay,0.1,0.6)
                    self.front_axle_idx, self.error_front_axle, self.front_axle_xy = mpc_helper.find_front_axle_idx(self.state, self.x_pred, self.y_pred, self.total_delay)
                    self.front_axle_idx = clip(self.front_axle_idx, 0, T-1)

                    if self.scenario == 'LANECHANGE':
                        MAX_DSTEER_lanechange = np.deg2rad(5.0/4.5)  # maximum steering speed [rad/s]
                        #all_steer_rate = clip(self.odelta, -MAX_DSTEER_lanechange, MAX_DSTEER_lanechange)
                        self.odelta = clip(self.odelta, -MAX_DSTEER_lanechange, MAX_DSTEER_lanechange)
                    else:
                        #all_steer_rate = clip(self.odelta, -MAX_DSTEER, MAX_DSTEER)
                        self.odelta = clip(self.odelta, -MAX_DSTEER, MAX_DSTEER)

                    #all_steer_rate = clip(self.odelta, -MAX_DSTEER, MAX_DSTEER)



                    all_steer_val = []
                    for si in range(len(self.odelta)):
                        #all_steer_val.append(self.state.delta + np.sum(all_steer_rate[:si+1] * self.DT))
                        all_steer_val.append(self.state.delta + np.sum(self.odelta[:si+1] * self.DT))

                    all_steer_val = np.array(all_steer_val)
                    all_steer_val = clip(all_steer_val, -MAX_STEER, MAX_STEER)
                    print('All Steer-Cmd: ' + str(np.round(-(degrees(all_steer_val)))) + " degs ")

                    #print('FRONT AXLE CMD  : ' + str(self.front_axle_idx) + "  Steer cmd: " + str(np.round(-(degrees(all_steer_val[self.front_axle_idx])))) + " degs " + str(self.target_ind))

                    self.last_di = clip(all_steer_val[self.front_axle_idx], -MAX_STEER, MAX_STEER)
                    self.steering_cmd_arr.append(self.last_di)
                    if len(self.steering_cmd_arr) > self.steer_filter_window:
                        self.steering_cmd_arr = self.steering_cmd_arr[-self.steer_filter_window:]

                    ## Filtering the steer command
                    sampling_freq = 10.0
                    filter_order = 2
                    cutoff = 1.8 # speed based dynamic cutoff frequency of the filte
                    self.last_di = tfil.butter_lowpass_filter(self.steering_cmd_arr, cutoff, sampling_freq, filter_order)[-1]

                    

                    # ## Taking commands after some delay
                    # cog_axle_idx_delay, _ = mpc_helper.find_front_axle_idx(self.state, self.x_pred, self.y_pred, 0.0, 1.5)
                    # cog_axle_idx_delay = clip(cog_axle_idx_delay, 0, T-1)
                    # all_steer_val_delay = []
                    # for si in range(len(self.odelta)):
                    #     if si >= cog_axle_idx_delay:
                    #         all_steer_val_delay.append(self.state.delta + np.sum(self.odelta[cog_axle_idx_delay:si+1] * self.DT))
                    #     else:
                    #         all_steer_val_delay.append(self.state.delta)
                    # all_steer_val_delay = clip(np.array(all_steer_val_delay), -MAX_STEER, MAX_STEER)
                    # self.last_di_2 = clip(all_steer_val_delay[self.front_axle_idx], -MAX_STEER, MAX_STEER)

                    #print('Commands     Steer-Rate: ' + str(np.round(degrees(self.odelta),1)) + " degs/sec ")

                    

                    #all_steer_val = self.state.delta + all_steer_rate * self.DT
                    
                    
                    #print('All Steer-Cmd: ' + str(np.round(-(degrees(all_steer_val) / self.steer_ratio))) + " degs ")


                    

                    
                    
                else:
                    print('NO Commands')




            
                # # warm-up solver
                # if True: #target_ind < 10:
                #     if abs(self.state.v) < 0.01:
                #         if self.path_speed_profile[self.target_ind]<0:
                #             self.last_ai = -0.01
                #         else:
                #             self.last_ai =  0.01


                if self.ego_brake_val >= 4.0 and self.throttle_gas_level > 0.0 and SIMULATION == False:
                    self.driver_brake = True
                    print(
                        "Safety Driver is BRAKING   ....  BRAKE %: "
                        + str(self.ego_brake_val),
                        throttle_duration_sec=1
                    )
                    print(
                        "PID control on Standby", 
                        throttle_duration_sec=1
                    )

                else:
                    self.driver_brake = False

                if self.driver_brake:
                    self.throttle_gas_level = 0.0
                    # self.throttle_gas_level = self.standby_max_brake
                    print(
                        "MPC Speed Control DeActivate Conditions: " 
                        + str(self.ego_vel < 5.0/3.6) + " " + str(self.driver_brake), 
                        throttle_duration_sec=1
                    )

                    self.I_term_prev = 0.0
                else:
                    print(
                        "MPC Control Node is Running in Decoupled Mode ...",
                        # throttle_duration_sec=1
                    )

                    #self.throttle_gas_level = cmd_conv.KS_acceleration_cmd_2_throttle_brake_cmd(self.last_ai, self.state.v, 0.0)

                    ## Speed target adjusted based on steer_error

                    lookahead_time = 0.8 #secs
                    lookahead_speed_target_idx = self.target_ind + int(clip(self.ego_vel*lookahead_time, 3.2, 11.0)/self.path_resolution)
                    lookahead_speed_target_idx = min(len(self.path_speed_profile)-1, lookahead_speed_target_idx)

                    ##self.speed_profile_target = self.path_speed_profile[self.target_ind]
                    self.speed_profile_target = self.path_speed_profile[lookahead_speed_target_idx]

                    self.speed_target_mod = min(
                        mpc_helper.steer_err_based_target_speed(self.last_di),
                        self.speed_profile_target,
                    )
                    print("LATDIST******************** " + str(self.lat_dist) + " " + str(mpc_helper.lat_error_based_target_speed(self.lat_dist)*3.6) + " kph")
                    print("TARGETSPEED******************** " + str(round(self.speed_profile_target*3.6,2)) + " kph  "+ str(round(self.speed_target_mod * 3.6,2)) + " kph    LEN " + str(self.path_length) + " " + str(self.target_ind))

                    if self.path_length < 15.0 and self.speed_target_mod == 0.0:
                        self.speed_target_mod = 0.0
                        self.throttle_gas_level = self.standby_max_brake
                        self.I_term_prev = 0.0
                    else:
                        ## PID Speed Control
                        self.throttle_gas_level, self.I_term_prev = pid_speed.PID_main(
                            self.speed_target_mod,
                            self.ego_vel,
                            self.I_term_prev,
                            self.direction_of_travel,
                            self.DT,
                            self.Kp_v,
                            self.tauI_v,
                        )


                ## Constraint the throttle commands
                self.last_throttle = pid_speed.constraint_throttle(
                    self.throttle_gas_level,
                    self.last_throttle,
                    self.DT,
                    self.max_throttle_change_accel,
                    self.max_throttle_change_accel_release,
                    self.max_throttle_change_brake,
                    self.max_throttle_change_brake_release,
                )

            else:
                print(
                    "MPC Control Node is in Standby Mode", 
                    throttle_duration_sec=1
                )

                ## If no path info available or localization health error  (brake 25% & bring steering to 0 degrees)
                self.state = mpc_helper.State(
                    x=self.ego_x, y=self.ego_y, yaw=self.ego_yaw, v=self.ego_vel, dt=self.DT
                )
                self.throttle_gas_level = self.standby_max_brake  # brake
                self.last_throttle = pid_speed.constraint_throttle(
                    self.throttle_gas_level,
                    self.last_throttle,
                    self.DT,
                    self.max_throttle_change_accel,
                    self.max_throttle_change_accel_release,
                    self.max_throttle_change_brake,
                    self.max_throttle_change_brake_release,
                )
                self.last_di = mpc_helper.constraint_di(
                    0.0,
                    self.last_di,
                    self.DT,
                    self.steer_ratio,
                    self.standby_max_steer_rate,
                )

            if SHOW_ANIMATION:
                self.plot_mpc(self.x_pred, self.y_pred)

        except Exception as e:
            print(
                "MPC Main Function Error : " + str(e) + str(traceback.format_exc())
            )
        
        self.last_throttle = self.ego_throttle_val
        self.last_steer = self.ego_steer_val
        self.last_brake = self.ego_brake_val