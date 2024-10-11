import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import agents.navigation.frenet.frenet_polynomials as fp_class


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, fp_params):
    """
    Calculates several frenet paths under the given iterables (lane_width & lane width division, Prediction Time & Time intervals)
    s0  # current course position
    c_speed # current speed [m/s]
    c_d  # current lateral position [m]
    c_d_d  # current lateral speed [m/s]
    c_d_dd  # current lateral acceleration [m/s]
    
    """

    frenet_paths = []

    # generate path to each offset goal
    for fp_param in fp_params:

        fp = fp_class.FrenetPath()
        fp.t = np.linspace(0.0, fp_param[2]-fp_param[3], int(fp_param[2]/fp_param[3])).tolist()

        # Lateral motion planning
        lat_qp = fp_class.QuinticPolynomial(c_d, c_d_d, c_d_dd, fp_param[1], 0.0, 0.0, fp_param[2])
        fp.d, fp.d_d, fp.d_dd, fp.d_ddd = lat_qp.calc_path_derivatives(fp_param[3])

        # Longitudinal motion planning
        lon_qp = fp_class.QuarticPolynomial(s0, c_speed, 0.0, fp_param[0], 0.0, fp_param[2])
        fp.s, fp.s_d, fp.s_dd, fp.s_ddd = lon_qp.calc_path_derivatives(fp_param[3])

        ## Cost calculation of the frenet path
        J_d = np.sum(np.array(fp.d_ddd) ** 2)  # sum of squares of lateral jerk
        J_s = np.sum(np.array(fp.s_ddd) ** 2)  # sum of squares of longitudinal jerk
        J_lat_e = np.sum(np.array(fp.d) ** 2)  # sum of squares of lateral deviation from reference path

        # square of diff from target speed
        ds = (fp_param[0] - fp.s_d[-1]) ** 2

        # cost weights
        K_J = 0.1
        K_D = 1.0
        K_LAT_err = 1.0

        K_LAT = 1.0
        K_LON = 1.0


        fp.cd = K_J * J_d + K_D * fp.d[-1] ** 2 + K_LAT_err * J_lat_e
        fp.cv = K_J * J_s + K_D * ds
        fp.cf = K_LAT * fp.cd + K_LON * fp.cv

        frenet_paths.append(fp)

    return frenet_paths


def calc_global_paths(fplist, tx, ty, tyaw):
    """
    Converts the Frenet Paths into Cartesian Coordinate Form
    """
    dx1 = np.append([0.0], np.diff(tx))
    dy1 = np.append([0.0], np.diff(ty))
    sx = np.cumsum(np.sqrt(dx1**2 + dy1**2))

    for fp in fplist:
        # calc global positions
        for i in range(len(fp.s)):
            traj_i = np.argmin(np.abs(sx - fp.s[i]))
            px_i = tx[traj_i]
            py_i = ty[traj_i]
            pyaw_i = tyaw[traj_i]

            di = fp.d[i]
            fx = px_i + di * np.cos(pyaw_i + np.pi / 2.0)
            fy = py_i + di * np.sin(pyaw_i + np.pi / 2.0)

            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))

        fp.yaw.append(fp.yaw[-1])

    return fplist


def get_car_outline(x, y, yaw):
    """
    Returns the Polygon of the Ego Vehicle based on the Dimensions, Given Input Ego Pose
    """
    # EQV/EVito Dimensions
    LENGTH = 5.140  # [m]
    WIDTH = 2.23  # [m]
    BACKTOWHEEL = 1.045  # [m] rear axle center to the back of the car
    outline = np.array(
        [
            [
                -BACKTOWHEEL,
                (LENGTH - BACKTOWHEEL),
                (LENGTH - BACKTOWHEEL),
                -BACKTOWHEEL,
                -BACKTOWHEEL,
            ],
            [WIDTH / 2, WIDTH / 2, -WIDTH / 2, -WIDTH / 2, WIDTH / 2],
        ]
    )

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])

    outline = (outline.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y

    return outline.T



def check_collision_dynamic(fplist, dynamic_ob, ego_x, ego_y):
    """
    Verifies all available paths with predicted dynamic obstacle reachability sets
    """
    for j, _ in enumerate(fplist):
        ## Initialize with False in the collision Flag
        fplist[j].dynamic_collision = [False] * len(fplist[j].x)
        fplist[j].dynamic_intersection_area = [0.0] * len(fplist[j].x)

        for i in range(len(fplist[j].x)):
            ego_poly = get_car_outline(fplist[j].x[i], fplist[j].y[i], fplist[j].yaw[i])
            ego_polygon = Polygon(ego_poly)
            
            ## Iterate over each obstacles
            for ob_i in range(len(dynamic_ob)):
                ## Iterate over dynamic obstacle polygons
                if len(dynamic_ob[ob_i][0]):
                    ego_poly = get_car_outline(
                        fplist[j].x[i], fplist[j].y[i], fplist[j].yaw[i]
                    )
                    ego_polygon = Polygon(ego_poly)
                    dynamic_obj_polygon = Polygon(np.array([dynamic_ob[ob_i][0], dynamic_ob[ob_i][1]]).T)
                    collision_flag = ego_polygon.intersects(dynamic_obj_polygon)
                    #collision_poly = ego_polygon.intersection(dynamic_obj_polygon)

                    fplist[j].dynamic_collision[i] = (
                        fplist[j].dynamic_collision[i] or collision_flag
                    )

                    #fplist[j].dynamic_intersection_area[i] = (
                    #    collision_poly.area
                    #)
        fplist[j].cf += np.sum(fplist[j].dynamic_intersection_area)
    return fplist



def check_dynamic_collision(fp, fp_path_time, fp_path_time_tick):
    time_of_path = 0.0
    for i in range(len(fp.dynamic_collision)):
        if fp.dynamic_collision[i] == False:
            time_of_path += fp_path_time_tick
        else:
            break

    if time_of_path > 3.0:
        return True
    else:
        return False


def check_paths(fplist, fp_path_time, fp_path_time_tick):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if not check_dynamic_collision(fplist[i], fp_path_time, fp_path_time_tick):
            print('Path ID: ' + str(i) + " eliminated due to Dynamic Obstacle")
            continue
        elif fplist[i].s[-1] < 1.5:  # PathLength check
            print('Path ID: ' + str(i) + " eliminated due to path_length being too short")
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def plot_fp(fp_list):
    """
    Plots all the calculated frenet paths
    """
    for i, _ in enumerate(fp_list):
        plt.plot(fp_list[i].x, fp_list[i].y, "-.b", lw=1.3)


def pi_2_pi(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def normalize_angle_arr(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    angle = np.array(angle)
    sin_val = np.sin(angle)
    cos_val = np.cos(angle)
    normalized_angle = np.arctan2(sin_val,cos_val)
    return normalized_angle


def filter_non_consistent_path_points(self, x_arr, y_arr, path_res):
    hypot_dist = np.hypot(np.diff(x_arr),np.diff(y_arr))

    index_arr = np.where(hypot_dist > 4*path_res)[0]

    #self.get_logger().debug('hypot ref: ' + str(hypot_dist) + " " + str(index_arr))
    if len(index_arr):
        return index_arr[0]
    else:
        return len(x_arr)-1


def get_last_idx(self, start_i, x, y, l_dist, r_dist, lane_width, goal_x, goal_y, path_res):
    """
    Calculates the index of the last point on the path array which indicates the end of free space
    Also corrects theis index incase the goal index is closer to the ego vehicle than the actual index of the last free point
    """
    if start_i < len(x):
        x_path = x[start_i:]
        y_path = y[start_i:]

        end_index_1 = filter_non_consistent_path_points(self, x_path, y_path, path_res)

        goal_i = np.argmin(np.hypot(x_path - goal_x, y_path - goal_y)) + start_i

        try:
            goal_dist = np.min(np.hypot(x_path - goal_x, y_path - goal_y))
        except:
            goal_dist = 1000.0

        l_path = np.clip(np.abs(l_dist[start_i:]), 0.0, lane_width/2.0)
        r_path = np.clip(np.abs(r_dist[start_i:]), 0.0, lane_width/2.0)
        total_dist_path = l_path + r_path

        #self.get_logger().debug(str(start_i) + "  Total_distance" + str(np.round(total_dist_path,1)))

        '''
        Ignore 2 consecutive points
        '''
        if len(total_dist_path) > 5:
            b = total_dist_path[:-2]
            c = total_dist_path[1:-1]
            d = total_dist_path[2:]
            e = np.maximum(b,c)
            total_dist_path = np.maximum(e,d)

        '''
        Ignore 3 consecutive points
        '''
        # if len(total_dist_path) > 5:
        #     b = total_dist_path[:-3]
        #     c = total_dist_path[1:-2]
        #     d = total_dist_path[2:-1]
        #     e = total_dist_path[3:]
        #     f = np.maximum(b,c)
        #     g = np.maximum(f,d)
        #     total_dist_path = np.maximum(g,e)

        #self.get_logger().debug(str(start_i) + "  Total_distance" + str(np.round(total_dist_path,1)))

        #l_flag = l_path >= lane_width / 3.0
        #r_flag = r_path >= lane_width / 3.0
        #comb_flag = np.logical_and(l_flag, r_flag)
        #false_idx = np.where(comb_flag == False)[0]

        total_flag = total_dist_path >= lane_width / 1.2
        false_idx = np.where(total_flag == False)[0]

        if len(false_idx) > 0:
            end_i = start_i + false_idx[0]
        else:
            end_i = start_i + len(x_path) - 1

        #self.get_logger().debug('GET_last_idx000: ' + str(end_i) + " " + str(end_index_1 + start_i))
        #self.get_logger().debug("  Total_distance FLAG  " + str(total_flag))

        end_i = min(end_index_1 + start_i, end_i)


        self.get_logger().debug('GET_last_idx: ' + str(end_i) + " " + str(false_idx))

        if end_i >= goal_i and goal_dist <= 2.0:
            return goal_i
        else:
            return end_i

    else:
        return start_i


def get_last_idx_2(self, start_i, x, y, l_dist, r_dist, lane_width, goal_x, goal_y):
    """
    Calculates the index of the last point on the path array which indicates the end of free space
    Also corrects theis index incase the goal index is closer to the ego vehicle than the actual index of the last free point
    """
    if start_i < len(x):
        x_path = x[start_i:]
        y_path = y[start_i:]

        goal_i = np.argmin(np.hypot(x_path - goal_x, y_path - goal_y)) + start_i

        try:
            goal_dist = np.hypot(x_path[goal_i-start_i] - goal_x, y_path[goal_i-start_i] - goal_y)
        except:
            goal_dist = 1000.0

        l_path = np.clip(np.abs(l_dist[start_i:]), 0.0, lane_width/2.0)
        r_path = np.clip(np.abs(r_dist[start_i:]), 0.0, lane_width/2.0)
        total_dist_path = l_path + r_path
        #self.get_logger().debug(str(start_i) + "  " + str(np.round(total_dist_path,1)))

        #l_flag = l_path >= lane_width / 3.0
        #r_flag = r_path >= lane_width / 3.0
        #comb_flag = np.logical_and(l_flag, r_flag)
        #false_idx = np.where(comb_flag == False)[0]

        total_flag = total_dist_path >= lane_width / 1.2
        false_idx = np.where(total_flag == False)[0]

        if len(false_idx) > 0:
            end_i = start_i + false_idx[0]
        else:
            end_i = start_i + len(x_path) - 1

        #self.get_logger().debug('GET_last_idx: ' + str(end_i) + " " + str(false_idx) + " " + str(goal_i) + " " + str(goal_dist) + str(total_flag))

        if end_i >= goal_i and goal_dist <= 2.0 and (len(x_path) - 1 - goal_i) < 10:
            return goal_i, True
        else:
            return end_i, False

    else:
        return start_i, False

def get_ego_loc_wrt_ref(ref_x, ref_y, ref_yaw, ego_x, ego_y, idx, long=False):
    """
    Calculate lateral distance to the ref_path & also the direction (which side)
    """
    if idx == 0:
        idx_1 = idx
        idx_2 = idx + 1
    else:
        idx_1 = idx - 1
        idx_2 = idx

    if long:
        # get idx closest to ego vehicle 
        return np.argmin(np.hypot(ref_x - ego_x, ref_y - ego_y)) 

    p1 = np.array([ref_x[idx_1], ref_y[idx_1]])
    p2 = np.array([ref_x[idx_2], ref_y[idx_2]])
    p3 = np.array([ego_x, ego_y])

    return np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)

def comfortable_stopping_distance(v):
    '''
    This function finds the stopping distance considering the speed of the vehicle and reaction time of the driver.
    '''
    t_1 = 1.5 ## Thinking time
    t_2 = 0.3 ## reaction time
    t_3 = 0.2 ## Brake effectiveness time
    f = 0.8 ## Friction of the road
    s = 0 ## slope of the road
    g = 9.8 ## gravity

    min_dist = 10.0
    max_dist = 50.0

    safety_distance = (v*3.6)/2.0   ## Half the speed is the safety distance
    s_D = (v * (t_1 + t_2 + t_3)) + v**2/(2 * g * (f + s)) ## Stopping distance

    final_dist = np.clip(s_D + safety_distance, min_dist, max_dist)
    return final_dist


# def safe_speed_calc(dist_ahead, decel):
#     '''
#     This function finds the stopping distance considering the speed of the vehicle and reaction time of the driver.
#     '''
#     t_1 = 1.5 ## Thinking time
#     t_2 = 0.3 ## reaction time
#     t_3 = 0.2 ## Brake effectiveness time
#     f = 0.8 ## Friction of the road
#     s = 0 ## slope of the road
#     g = 9.8 ## gravity

#     min_dist = 10.0
#     max_dist = 50.0

#     safety_distance = (v*3.6)/2.0   ## Half the speed is the safety distance
#     s_D = (v * (t_1 + t_2 + t_3)) + v**2/(2 * g * (f + s)) ## Stopping distance

#     final_dist = np.clip(s_D + safety_distance, min_dist, max_dist)
#     return final_dist


# def safe_speed_calc(dist_ahead, decel, safety_distance):
#     '''
#     This function finds the safe speed given a distance & decelration#
#     '''
#     ##safety_distance = 17.0
#     dist_safe = max(dist_ahead-safety_distance, 0.0)
#     safe_speed = np.sqrt(2*abs(decel)*dist_safe)
#     return safe_speed


def safe_speed_calc(ego_vel, dist_ahead, decel, buffer_distance):
    '''
    This function finds the safe speed given a buffer distance (space available ahead) & deceleration 
    '''
    
    time_delay = 1.0 #secs
    
    reaction_distance = round(ego_vel*time_delay,1)
    
    speed_2_distance_gain = 1.0 ## Gain for Effective braking distance accounted based on current ego speed 
    #braking_distance = (ego_vel * 3.6) * speed_2_distance_gain
    braking_distance = (ego_vel**2) / (2 * decel)

    # Max & Min Distance Constraint on Total Distance available
    min_dist = 10.0 # mtrs
    max_dist = 70.0 # mtrs
    total_distance = np.clip(reaction_distance+braking_distance+buffer_distance, min_dist, max_dist)
    
    dist_safe = max(dist_ahead - reaction_distance - buffer_distance, 0.0)

    # Max & Min Speed Constraint on Safe Speed
    min_safe_speed = 0.0/3.6 # m/sec
    max_safe_speed = 50.0/3.6 # m/sec
    safe_speed = np.clip(dist_safe / decel, min_safe_speed, max_safe_speed)
    
    return safe_speed
    

def choose_ref_path(
    self,
    ego_x,
    ego_y,
    ego_yaw,
    ego_vel,
    px,
    py,
    pyaw,
    pcurv,
    p_ld,
    p_rd,
    p_idx,
    p_len_idx,
    p_priority,
    p_morph,
    p_reason,
    lane_width,
    path_resolution,
    goal_x,
    goal_y,
    choose_ref_decision_distance,
    prev_start_idx,
):
    """
    Chooses the ref path for frenet calculations among multiple lanes based on available distance ahead & closest lane
    """

    ## ****************TODO HIGH SPEED LANE CHANGES (30kph & above)

    if ego_vel > 30.0/3.6:
        choose_ref_decision_distance = 45.0


    ## Populate curcial information in the following arrays to help decide lane choice
    start_idx_arr = []
    route_idx_arr = []
    route_len_idx_arr = []
    lat_dist_arr = []
    end_idx_arr = []
    dist_available = []
    dist_available_back = []

    abrupt_change_arr = []
    abrupt_change_dist_arr = []
    lane_reason_arr = []

    for i in range(len(px)):
        hypot_i = np.hypot(np.array(px[i]) - ego_x, np.array(py[i]) - ego_y)

        #hypot_d = np.hypot(np.diff(px[i]), np.diff(py[i]))
        self.get_logger().debug("*****************Hypot_dist_lane_id: " + str(i) + "  " + str(min(hypot_i)) + "  " + str(np.argmin(hypot_i)))


        #print(i, np.round(np.degrees(pyaw[i][:20]),1))
        
        heading_chk_flag = np.abs(normalize_angle_arr(np.array(pyaw[i])-ego_yaw)) <  np.pi/2.0

        #self.get_logger().debug(str(i) + "  *****************Hypot_dist_angle_chk: " + str(hypot_i) + "  " + str(normalize_angle_arr(np.array(pyaw[i])-ego_yaw)))




        hypot_flag_i = hypot_i[heading_chk_flag]

        if len(hypot_flag_i):
            ref_start_idx_arr = np.where(heading_chk_flag)[0]
            ref_start_idx = np.argmin(hypot_flag_i)

            ref_start_dist_sorted_idx = np.argsort(hypot_flag_i)
            idxs = ref_start_idx_arr[ref_start_dist_sorted_idx]

            dist_2_closest_pt = np.min(hypot_flag_i)

            if dist_2_closest_pt > 10.0:
                start_idx = 0
                lat_dist = 1000.0
                end_idx = 0
                route_idx = 0
            else:
                # if len(prev_start_idx) == 0:
                #     start_idx = ref_start_idx_arr[ref_start_idx]
                # else:
                #     prev_idx = prev_start_idx[i]
                #     for id_i in range(len(idxs)):
                #         if abs(idxs[id_i] - prev_idx) < 20:
                #             start_idx = idxs[id_i]
                #             break

                start_idx = ref_start_idx_arr[ref_start_idx]

                lat_dist = hypot_i[start_idx]
                route_idx = p_idx[i][start_idx]
                end_idx = get_last_idx(
                    self, start_idx, px[i], py[i], p_ld[i], p_rd[i], lane_width, goal_x, goal_y, path_resolution
                )
        else:
            start_idx = 0
            lat_dist = 1000.0
            end_idx = 0
            route_idx = 0

        self.get_logger().debug("***************************start_idx00: " + str(start_idx))


        '''
        start_idx = np.argmin(hypot_i)
        lat_dist = np.min(hypot_i)
        '''
        
        start_idx_arr.append(start_idx)
        lat_dist_arr.append(lat_dist)
        end_idx_arr.append(end_idx)
        route_idx_arr.append(route_idx)
        route_len_idx_arr.append(p_len_idx[i])
        lane_free_space = np.sum(np.hypot(np.diff(px[i][start_idx : end_idx]),np.diff(py[i][start_idx : end_idx])))
        lane_free_space_back = np.sum(np.hypot(np.diff(px[i][:start_idx]),np.diff(py[i][:start_idx])))
        lane_reason_arr.append(p_reason[i][-1])

        end_i_print = min(len(p_ld[i])-1,end_idx + 4)
        #self.get_logger().debug("*********LANE FREE DISTANCES  : " + str(i) + "  Left: "  + str(p_ld[i][start_idx : end_i_print]) + "   Right: " + str(p_rd[i][start_idx : end_i_print]))
        
        path_yaw_diff = np.abs(normalize_angle_arr(np.diff(get_heading_arr(np.array(px[i]), np.array(py[i])))))

        #self.get_logger().debug("*********PATH_YAWDIFF: " + str(i) + " " + str(np.round(np.degrees(path_yaw_diff),1)))

        path_yaw_abrupt_idx = np.where(path_yaw_diff > np.radians(75.0))[0]
        #self.get_logger().debug("*********PATH_YAW Abrupt index: " + str(path_yaw_abrupt_idx))
        #self.get_logger().debug("*********PATH_YAW Abrupt: " + str(np.round(np.degrees(path_yaw_diff),1)))

        if len(path_yaw_abrupt_idx) > 0:
            path_yaw_abrupt = True
            idx_0 = path_yaw_abrupt_idx[0]

            if start_idx < idx_0:
                path_yaw_abrupt_dist = np.sum(np.hypot(np.diff(px[i][start_idx : idx_0]),np.diff(py[i][start_idx : idx_0])))
            else:
                path_yaw_abrupt_dist = 0.0
        else:
            path_yaw_abrupt = False
            path_yaw_abrupt_dist = lane_free_space

        dist_available.append(lane_free_space)
        dist_available_back.append(lane_free_space_back)
        abrupt_change_arr.append(path_yaw_abrupt)
        abrupt_change_dist_arr.append(path_yaw_abrupt_dist)


    if len(dist_available) > 1:
        dist_available_diff = np.diff(dist_available)
        if np.all(dist_available_diff < 4.0):
            dist_available = np.min(dist_available) * np.ones(len(dist_available))
            dist_available[1] = dist_available[1]-5.1


    self.get_logger().debug("*********LAT dist arr: " + str(lat_dist_arr))
    ## Rearranging in ascending order based on the lat_dist of each paths
    index_ascending = np.argsort(lat_dist_arr)
    start_idx_arr_2 = np.array(start_idx_arr)[index_ascending]
    lat_dist_arr = np.array(lat_dist_arr)[index_ascending]
    end_idx_arr = np.array(end_idx_arr)[index_ascending]
    route_idx_arr = np.array(route_idx_arr)[index_ascending]
    route_len_idx_arr = np.array(route_len_idx_arr)[index_ascending]
    #path_prio_idx_arr = np.array(path_prio_idx_arr)[index_ascending]
    p_priority = np.array(p_priority)[index_ascending ]
    dist_available = np.array(dist_available)[index_ascending]
    dist_available_back = np.array(dist_available_back)[index_ascending]
    abrupt_change_arr = np.array(abrupt_change_arr)[index_ascending]
    abrupt_change_dist_arr = np.array(abrupt_change_dist_arr)[index_ascending]
    lane_reason_arr = np.array(lane_reason_arr)[index_ascending]



    chosen_lane_idx_arr = []
    path_prio_arr = []
    for lane_i in range(len(start_idx_arr_2)):
        # ego_dist_available_lane_i = (
        #     end_idx_arr[lane_i] - start_idx_arr_2[lane_i]
        # ) * path_resolution
        ego_dist_available_lane_i = dist_available[lane_i]

        self.get_logger().debug("Start & End Index Arr: " + str(start_idx_arr_2) + " " + str(end_idx_arr))
        self.get_logger().debug("*********Available in lane: " + str(lane_i) + " "  + str(p_priority[lane_i]) + " " + str(ego_dist_available_lane_i) + "  Abrupt: " + str(abrupt_change_arr[lane_i]) + " Dist: " + str(abrupt_change_dist_arr[lane_i]))
        self.get_logger().debug("*********Available in back lane: " + str(lane_i) + " "  + str(dist_available_back[lane_i]))
        ## Condition to factor abrupt change in yaw
        #condition_abrupt = (abrupt_change_arr[lane_i] == False) and (np.min(np.diff(dist_available)) < 10)


        if len(dist_available) > 1:
            if np.min(np.diff(dist_available)) < 10:
                # if lane_i == 0:
                #     available_dist_greater_flag = abrupt_change_dist_arr[0] > abrupt_change_dist_arr[1]
                # else:
                #     available_dist_greater_flag = abrupt_change_dist_arr[1] > abrupt_change_dist_arr[0]

                cond_1 = abrupt_change_arr[lane_i] == False
                #cond_2 = (abrupt_change_dist_arr[lane_i] < choose_ref_decision_distance) or available_dist_greater_flag
                condition_abrupt = cond_1# and cond_2
            else:
                condition_abrupt = True
        else:
            condition_abrupt = True
        #self.get_logger().debug("***********************Abrupt: "  + str(condition_abrupt))
        #self.get_logger().debug("***********************YIELD FLAG: "  + str(not('yield_sign' in lane_reason_arr)))


        #if ego_dist_available_lane_i > comfortable_stopping_distance(ego_vel):
        if ego_dist_available_lane_i > choose_ref_decision_distance and condition_abrupt and not('yield_sign' in lane_reason_arr):
            chosen_lane_idx_arr.append(index_ascending[lane_i])
            path_prio_arr.append(p_priority[lane_i])

    self.get_logger().debug("***********************Chosenlaneidx 0 : " + str(chosen_lane_idx_arr))

    if len(chosen_lane_idx_arr) == 0 and len(dist_available) and not('yield_sign' in lane_reason_arr):
        idx = np.argmax(dist_available)
        if len(dist_available) > 1:
            dist_diff = np.diff(dist_available)
            if dist_diff[0] > 10.0:
                chosen_lane_idx_arr.append(index_ascending[idx])
                path_prio_arr.append(p_priority[idx])


    self.get_logger().debug("***********************Chosenlaneidx 1 : " + str(chosen_lane_idx_arr) + str(path_prio_arr))




    if len(chosen_lane_idx_arr) == 0:
        x_lane = px[index_ascending[0]]
        y_lane = py[index_ascending[0]]
        yaw_lane = pyaw[index_ascending[0]]
        curv_lane = pcurv[index_ascending[0]]
        left_distance = p_ld[index_ascending[0]]
        right_distance = p_rd[index_ascending[0]]
        lane_morphology = p_morph[index_ascending[0]]
        lane_reason = p_reason[index_ascending[0]]
        route_percentage = route_idx_arr[0]/route_len_idx_arr[0]
        final_start_idx = start_idx_arr_2[0]
        final_end_idx = end_idx_arr[0]
        self.get_logger().debug("******************************************Chosen 1: " + str(index_ascending[0]) + " " + str(index_ascending))
        chosen_lane_idx = index_ascending[0]

        if len(start_idx_arr_2) > 1:
            final_start_idx_1 = start_idx_arr_2[1]
            final_end_idx_1 = end_idx_arr[1]
        else:
            final_start_idx_1 = 0
            final_end_idx_1 = 0
        
    else:
        priority_order = np.argsort(path_prio_arr)
        self.get_logger().debug("****************************PathPriority: " + str(priority_order) + " " + str(path_prio_arr) + " " + str(p_priority))
        chosen_lane_idx = chosen_lane_idx_arr[priority_order[0]]
        if len(priority_order) > 1:
            chosen_lane_idx_adjacent = chosen_lane_idx_arr[priority_order[1]]
            final_start_idx_1 = start_idx_arr_2[chosen_lane_idx_adjacent]
            final_end_idx_1 = end_idx_arr[chosen_lane_idx_adjacent]

        else:
            final_start_idx_1 = 0
            final_end_idx_1 = 0
        #print('Chosen_lane',chosen_lane_idx)
        x_lane = px[chosen_lane_idx]
        y_lane = py[chosen_lane_idx]
        #final_start_idx = start_idx_arr_2[chosen_lane_idx]
        idx_option = p_priority[chosen_lane_idx]
        lane_morphology = p_morph[chosen_lane_idx]
        lane_reason = p_reason[chosen_lane_idx]
        final_start_idx = start_idx_arr_2[idx_option]
        final_end_idx = end_idx_arr[idx_option]

        self.get_logger().debug("******************************************Chosen 2: " + str(chosen_lane_idx))
        self.get_logger().debug("***************************start_idx01: " + str(final_start_idx))

        yaw_lane = pyaw[chosen_lane_idx]
        curv_lane = pcurv[chosen_lane_idx]
        left_distance = p_ld[chosen_lane_idx]
        right_distance = p_rd[chosen_lane_idx]
        route_percentage = route_idx_arr[chosen_lane_idx]/route_len_idx_arr[chosen_lane_idx]

    return x_lane, y_lane, yaw_lane, curv_lane, left_distance, right_distance, lane_morphology, lane_reason, route_percentage, final_start_idx, final_end_idx, final_start_idx_1, final_end_idx_1, chosen_lane_idx, start_idx_arr


def plot_arrow(x, y, yaw, length=2.0, width=1.2, fc="r", ec="k"):
    if isinstance(x, list):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(
            x,
            y,
            length * math.cos(yaw),
            length * math.sin(yaw),
            fc=fc,
            ec=ec,
            head_width=width,
            head_length=width,
        )
        plt.plot(x, y)


def get_normals(x, y, length_left, length_right):
    "Plotting the normals to the left and right for visualization"
    # fig, ax = plt.subplots()
    if len(length_left):
        for idx in range(len(x) - 1):
            x0, y0, xa, ya = x[idx], y[idx], x[idx + 1], y[idx + 1]
            dx, dy = xa - x0, ya - y0
            norm = math.hypot(dx, dy) * 1 / length_left[idx]
            dx /= norm
            dy /= norm
            plt.plot((x0, x0 - dy), (y0, y0 + dx), c="g", lw="0.5")  # plot the normals
    if len(length_right):
        for idx in range(len(x) - 1):
            x0, y0, xa, ya = x[idx], y[idx], x[idx + 1], y[idx + 1]
            dx, dy = xa - x0, ya - y0
            norm = math.hypot(dx, dy) * 1 / length_right[idx]
            dx /= norm
            dy /= norm
            plt.plot((x0, x0 - dy), (y0, y0 + dx), c="r", lw="0.5")


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    :param x: (float)
    :param y: (float)
    :param z: (float)
    :param w: (float)
    :return: (float, float, float)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def get_heading_arr(x_arr, y_arr):
    """
    Calculate heading of the input path
    """
    if len(x_arr) > 1:
        yaw_arr = np.arctan2(np.diff(y_arr), np.diff(x_arr))
        yaw_arr = np.append(yaw_arr, yaw_arr[-1])
    else:
        yaw_arr = np.array([])
    return yaw_arr.tolist()


def RotateAxis(AnchorX, AnchorY, inputx, inputy, WindDirection):
    """
    Rotates a polygon by the given angle around the input anchor point
    :param AnchorX: (float)
    :param AnchorX: (float)
    :param inputx: (float arr)
    :param inputy: (float arr)
    :param WindDirection: (float)
    :return: (float arr, float arr)
    """

    x = inputx - AnchorX
    y = inputy - AnchorY
    resultx = (x * math.cos(WindDirection)) - (y * math.sin(WindDirection)) + AnchorX
    resulty = (x * math.sin(WindDirection)) + (y * math.cos(WindDirection)) + AnchorY
    return (resultx.tolist(), resulty.tolist())
