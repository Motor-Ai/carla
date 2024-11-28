import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import os
# act utils imports
import agents.navigation.frenet.frenet_utils as fp_helper
import agents.navigation.frenet.cubic_spline_planner as cp


def get_frenet_traj_sequential(self, map_lanes, ego_pos, curr_speed_SU):
    maneuvers = ["KS", "AC", "DC", "LCL", "LCR"]
    maneuver_colors = ["red", "blue", "green", "purple", "brown", "pink"]

    try:
        os.mkdir("./decide_videos")
        os.mkdir("./decide_figures")
    except:
        pass

    plt.figure()

    for sample in range(1): # > 1 for multiple samples
        # execute sequential maneuvers
        ego_pos_planned = ego_pos.clone()
        ego_vel_planned = curr_speed_SU
        long_term_frenet = []
        best_long_term_maneuver = []
        for col in ["red", "blue", "green"]:
            duration = random.randint(30, 30) # choose random duration
            frenet_trajectories, ego_vels_planned = get_frenet_traj(self, map_lanes, ego_pos_planned, ego_vel_planned, duration=duration)
            best_maneuver = random.randint(0,len(frenet_trajectories)-1) # choose random maneuver
            best_maneuver_name = maneuvers[best_maneuver]
            best_trajectory = frenet_trajectories[best_maneuver]
            ego_vel_planned = ego_vels_planned[best_maneuver]
            ego_pos_planned = best_trajectory[-1]

            plt.plot(
                best_trajectory[:, 0].cpu().detach().numpy(),
                best_trajectory[:, 1].cpu().detach().numpy(),
                linewidth=3,
                color=col,
            )

            plt.text(
                best_trajectory[-1, 0].cpu().detach().numpy()+2,
                best_trajectory[-1, 1].cpu().detach().numpy()+2,
                maneuvers[best_maneuver]+f"( {ego_vel_planned:.2f} )",
                fontsize=8,
                color=col,
                zorder=10,
            )

            long_term_frenet.append(frenet_trajectories)
            best_long_term_maneuver.append(best_trajectory)

        for lane in map_lanes[0, 0]:
            plt.scatter(
                lane[:, 0].cpu().detach().numpy(),
                lane[:, 1].cpu().detach().numpy(),
                s=3,
            )

    best_long_term_maneuver = torch.cat(best_long_term_maneuver)

    FOV_x = min(max(30, best_trajectory[-1, 0].cpu().detach().abs().item() + 5), 200)
    FOV_y = min(max(30, best_trajectory[-1, 1].cpu().detach().abs().item() + 5), 200)
    FOV = max(FOV_x, FOV_y)
    plt.xlim(-FOV, FOV)
    plt.ylim(-FOV, FOV)
    #plt.show()
    plt.savefig(f"./decide_figures/z_longterm_frenet_egocentric_{time.time()}.png")
    plt.close()

    return long_term_frenet, best_long_term_maneuver


def get_reachable_lanes(local_env, lanes_ids_range=[-1, 1]):
    # TODO do in parallel and stitch succesive lanes for frenet
    reachable_lanes = []
    road_id_curr = local_env.lanes[0].rl_id.split("_")[0]
    for road_ids in [road_id_curr]+[rl.split("_")[0] for rl in local_env.lanes[0].successive_rl_ids]:
        print(road_ids)
        for lane_id in lanes_ids_range:
            print(lane_id)
            for lane in local_env.lanes:
                if lane.rl_id == f"{road_ids}_{lane_id}":
                    reachable_lanes.append(lane)

    return reachable_lanes


def get_frenet_traj(self, map_lanes, ego_pos, curr_speed_SU, duration=None):

    ego_pos = ego_pos.cpu().detach().numpy()

    # longitudinal distance / course progress # TODO is this correct?
    lon_params = [
        0,
        curr_speed_SU,
    ]  # longitudinal position and speed

    lat_dists = []
    long_dists = []
    map_lanes_temp = []
    for lane in map_lanes[0, 0, :]:
        # Make it unique to avoid issues with frenet calc
        lane =  torch.unique_consecutive(lane, dim = -2)
        # Consider only the non zero elements in the lane
        non_zero_map_elements = torch.any(lane[:, :2] != 0.0, dim=-1)
        nz = non_zero_map_elements.unsqueeze(-1).expand_as(lane)
        lane = torch.masked_select(lane, nz).view(-1, lane.shape[-1]).cpu().detach().numpy()

        x_lane = lane[:, 0]
        y_lane = lane[:, 1]
        lane_yaw = lane[:, 2]

        long_dist = np.argmin(np.hypot(x_lane - ego_pos[0], y_lane - ego_pos[1]))

        lat_dist = fp_helper.get_ego_loc_wrt_ref(
            x_lane,
            y_lane,
            lane_yaw,
            ego_pos[0],
            ego_pos[1],
            long_dist,
        )
        lat_dists.append(lat_dist)
        long_dists.append(long_dist)

        map_lanes_temp.append(lane)

    # sort map_lanes_nz and lat_dists based on lat_dists
    map_lanes_temp = [d[0]
            for d in sorted(
                zip(list(map_lanes_temp), lat_dists), key=lambda pair: pair[1]
            )]
        

    sorts = sorted(zip(long_dists, lat_dists), key=lambda pair: pair[1])
    long_dists =  [s[0] for s in sorts]
    lat_dists = [s[1] for s in sorts]

    # find current lane # TODO this will probably fail when adding more lanes from neighbors
    current_lane_id = np.argmin(np.abs(lat_dists))

    # find the left lane or fall back to the closest lane
    if current_lane_id > 0:
        left_lane_id = (
            current_lane_id - 1
        )  # left lane has lower lat_dist TODO improve this
    else:
        left_lane_id = current_lane_id

    # find the right lane or fall back to the closest lane
    if current_lane_id < len(lat_dists) - 1:
        right_lane_id = current_lane_id + 1
    else:
        right_lane_id = current_lane_id

    # go through attended lanes per maneuver ("KS", "AC", "DC") on current lane and ("LCL", "LCR") on left and right lanes
    fplist_all = []
    fp_manuvers = ["KS", "AC", "DC", "LCL", "LCR"]  # same order as in simulator
    ACCELERATION = (
        random.randint(-7, 7)  # TODO this value must fit the car and be matched in simulation
    )
    for i, closest_lane_id in enumerate(
        [
            current_lane_id,
            current_lane_id,
            current_lane_id,
            left_lane_id,
            right_lane_id,
        ]
    ):
        lat_dist = lat_dists[closest_lane_id]
        long_dist = long_dists[closest_lane_id]
        x_lane_current = (
            map_lanes_temp[closest_lane_id][:,0]
        )
        y_lane_current = (
            map_lanes_temp[closest_lane_id][:,1]
        )

        tx, ty, tyaw, _ = cp.calc_bspline_course(
            x_lane_current,
            y_lane_current,
            1.0,
            0.01,
        )

        lat_vel = -1 * np.sign(lat_dist) * 0.1 * (lat_dist**2)
        lat_vel_max = 1.5  # m/sec
        lat_vel = np.clip(lat_vel, -lat_vel_max, lat_vel_max)
        lat_accel_max = 0.55  # m/s²
        lat_accel_gain = 0.45 * (ego_pos[3] / 10.0)
        lat_accel = np.clip(
            -1 * np.sign(lat_dist**2) * lat_accel_gain,
            -lat_accel_max,
            lat_accel_max,
        )

        lat_params = [
            lat_dist,
            lat_vel,
            lat_accel,
        ]

        s0 = long_dist  # current course position
        c_speed = lon_params[1]  # current speed [m/s]
        c_d = lat_params[0]  # current lateral position [m]
        c_d_d = lat_params[1]  # current lateral speed [m/s]
        c_d_dd = lat_params[2]  # current lateral acceleration [m/s]

        fp_target_speeds_all = np.array(
            [
                max(c_speed, 1),  # KS
                c_speed + abs(ACCELERATION),  # ACC
                max(c_speed - abs(ACCELERATION), 1),  # DC
                max(c_speed + ACCELERATION, 1),  # LCL
                max(c_speed + ACCELERATION, 1),  # LCR
            ]
        )

        fp_lateral_dists = np.array([0, 0, 0, 0, 0])
        if duration is None:
            fp_plan_time = np.ones(5) * (self.len_fut * (1/self.predictor_fps))
        else:
            fp_plan_time = np.ones(5) * duration
        fp_plan_dt = np.ones(5) * (1/self.predictor_fps) 

        # select parameters for this maneuver
        fp_lateral_dists = fp_lateral_dists[i : i + 1]
        fp_plan_time = fp_plan_time[i : i + 1]
        fp_plan_dt = fp_plan_dt[i : i + 1]
        fp_target_speeds = fp_target_speeds_all[i : i + 1]
        fp_params = np.transpose(
            [fp_target_speeds, fp_lateral_dists, fp_plan_time, fp_plan_dt]
        )

        ## Calculate Frenet Paths w.r.t to the iterables
        fplist = fp_helper.calc_frenet_paths(
            c_speed, c_d, c_d_d, c_d_dd, s0, fp_params
        )

        ## Converts Paths from Frenet to Cartesian Coordinate System
        fplist = fp_helper.calc_global_paths(self, fplist, tx, ty, tyaw)

        fplist_all.append(fplist[0])
    fplist = fplist_all

    self.batch_size = 5  # No. of Manuvers

    if duration is None:
        frenet_points = self.len_fut
    else:
        frenet_points = duration
    frenet_trajectories = torch.zeros(
        [self.batch_size, frenet_points, 17], device=self.device
    )  

    manuver_laneid = {}
    for i, (traj, manuver) in enumerate(zip(fplist, fp_manuvers)):
        frenet_trajectories[i, ..., [0, 1, 2]] = torch.tensor(
            [traj.x, traj.y, traj.yaw], dtype=self.dtype, device=self.device
        ).T[:frenet_points, :]

        # Mapping Manuer class to LaneID
        last_traj_point = [traj.x[-1], traj.y[-1]]
        dists = []
        for lane in map_lanes_temp:
            last_lane_points = (
                lane[:, :2]
            )

            dists.append(
                np.min(np.linalg.norm(last_lane_points - last_traj_point, axis=1))
            )

        manuver_laneid[manuver] = np.argmin(np.array(dists))

    # plt.cla()
    # for lane in map_lanes_temp:
    #     plt.scatter(lane[:,0], lane[:,1])
    #     plt.scatter(lane[:,0], lane[:,1])
    #     # Plot the text of ID of the lane points
    #     # for i,(x,y) in enumerate(zip(lane[:,0],lane[:,1])):
    #     #     plt.text(x,y, str(i))
    #     plt.xlim(-5,70)
    #     plt.ylim(-100,100)

    # for traj in frenet_trajectories:
    #     plt.plot(traj[:,0].cpu(), traj[:,1].cpu())
    # plt.pause(0.01)

    if duration is not None:
        return frenet_trajectories, fp_target_speeds_all
    else:
        return frenet_trajectories