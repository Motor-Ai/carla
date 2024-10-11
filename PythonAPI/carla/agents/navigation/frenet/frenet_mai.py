import torch
import numpy as np
import agents.navigation.frenet.frenet_utils as fp_helper
import  agents.navigation.frenet.cubic_spline_planner as cp
import numpy as np
import shapely as sp
from shapely.affinity import affine_transform, rotate

def get_ego_velocity(ego):
    # assumes input is has shape [8], i.e. no batach dim, no time dim
    v_x = ego[3]
    v_y = ego[4]

    vel_ms = np.sqrt(v_x**2 + v_y**2)  # m/s
    vel_kmh = int(vel_ms * 3600 / 1000)
    # theta = np.arctan(v_y / v_x)

    # print(f"Velocity: {vel_ms} m/s, {vel_kmh} km/h, theta: {theta}")

    return vel_ms, vel_kmh

def egocentric_to_global(center, data_egocentric, velocity = True):
    data_global = data_egocentric.clone().detach()

    data_global[..., 0] = (
        center[0]
    + data_egocentric[..., 0] * torch.cos(center[2])
        - data_egocentric[..., 1] * torch.sin(center[2])
    )
    data_global[..., 1] = (
        center[1]
        + data_egocentric[..., 1] * torch.cos(center[2])
        + data_egocentric[..., 0] * torch.sin(center[2])
    )

    # heading to global # TODO double check
    data_global[..., 2] = data_egocentric[..., 2] + center[2]

    if velocity:
        # Velocity to global
        data_global[..., 3] = data_egocentric[..., 3] * torch.cos(
            center[2]
        ) - data_egocentric[..., 4] * torch.sin(center[2])
        data_global[..., 4] = data_egocentric[..., 4] * torch.cos(
            center[2]
        ) + data_egocentric[..., 3] * torch.sin(center[2])
    
    return data_global

def global_to_egocentric( center, data_global_, velocity=True):
    center = center.detach().cpu()
    data_global = data_global_.clone().detach()
    data_egocentric = torch.zeros_like(data_global)
    data_egocentric[..., 2:] = data_global[..., 2:]
    t_history = data_global.shape[-2]

    # history rereferenced to the current ego pose
    line = sp.LineString(
        [
            [x, y]
            for x, y in zip(
                data_global[0, :, 0].cpu().detach().numpy(),
                data_global[0, :, 1].cpu().detach().numpy(),
            )
        ]
    )
    line_offset = affine_transform(
        line, [1, 0, 0, 1, -center[0].cpu(), -center[1].cpu()]
    )
    line_rotate = rotate(line_offset, -center[2], origin=(0, 0), use_radians=True)
    line_rotate = np.array(line_rotate.coords)
    line_rotate[data_global[0, :, :2].cpu() == 0] = 0  # TODO why?

    heading = (
        data_global[0, :, 2].cpu() - center[2]
    )  # TODO wrap to pi # TODO plus angle or minus???

    heading[data_global[0, :, 2].cpu() == 0] = 0  # TODO why?
    data_egocentric[..., :2] = torch.tensor(
        line_rotate, dtype=data_egocentric.dtype, device=data_egocentric.device
    ).view(1, t_history, -1)
    data_egocentric[..., 2] = heading  # TODO construct new tensor instead?

    if velocity:
        # rereference ego velocity to the current ego pose
        data_g = data_global.clone().detach()
        data_egocentric[..., 3] = data_g[..., 3].cpu().detach() * np.cos(
            center[2]
        ) + data_g[..., 4].cpu().detach() * np.sin(
            center[2]
        )  # plus not minus !
        data_egocentric[..., 4] = data_g[..., 4].cpu().detach() * np.cos(
            center[2]
        ) - data_g[..., 3].cpu().detach() * np.sin(center[2])

    return data_egocentric.clone().detach().to(data_global_.device)

def get_frenet_traj(map_lanes, ego_pos):

        # # Consider only the non zero elements in the map
        # non_zero_map_elements = torch.any(map_lanes[..., :2] != 0.0, dim=-1)
        # # For both lanes
        # non_zero_map_elements = torch.logical_and(
        #     non_zero_map_elements[:, :, 0, :], non_zero_map_elements[:, :, 1, :]
        # )
        # nz = non_zero_map_elements.unsqueeze(-1).expand_as(map_lanes)
        
        # map_lanes_nz = torch.masked_select(map_lanes, nz).view(
        #     map_lanes.size(0),
        #     map_lanes.size(1),
        #     map_lanes.size(2),
        #     -1,
        #     map_lanes.size(4),
        # )
        map_lanes_nz = map_lanes

        x_lane = map_lanes_nz[0, 0, :, :, 0].cpu().detach().numpy()
        y_lane = map_lanes_nz[0, 0, :, :, 1].cpu().detach().numpy()
        lane_yaw = map_lanes_nz[0, 0, :, :, 2].cpu().detach().numpy()

        # ego_pos = ego_pos.cpu().detach().numpy()

        # longitudinal distance / course progress # TODO is this correct?
        lon_params = [
            0,
            5,
        #    get_ego_velocity(ego_pos)[0],
        ]  # longitudinal position and speed

        lat_dists = []
        long_dists = []
        for lane_id in range(map_lanes_nz.shape[2]):
            lat_dist = fp_helper.get_ego_loc_wrt_ref(
                x_lane[lane_id, :],
                y_lane[lane_id, :],
                lane_yaw[lane_id, :],
                ego_pos[0],
                ego_pos[1],
                1,
            )
            long_dist = fp_helper.get_ego_loc_wrt_ref(
                x_lane[lane_id, :],
                y_lane[lane_id, :],
                lane_yaw[lane_id, :],
                ego_pos[0],
                ego_pos[1],
                1,
                long=True,
            )
            lat_dists.append(lat_dist)
            long_dists.append(long_dist)

        # lat_dists = [-4, 0, 4] # We have only one lane
        # long_dists = [long_dist, long_dist, long_dist] 
        # map_lanes_nz = map_lanes_nz.repeat(1,1,len(lat_dists),1,1)

        # sort map_lanes_nz and lat_dists based on lat_dists

        map_lanes_nz[0, 0] = torch.stack(
            [
                d[0]
                for d in sorted(
                    zip(list(map_lanes_nz[0, 0]), lat_dists), key=lambda pair: pair[1]
                )
            ]
        )
        long_dists = [s[0] for s in sorted(zip(long_dists, lat_dists))]
        lat_dists = sorted(lat_dists)

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
        # TODO this re-uses the existing code from @mangal which used a single centerline, of which lot is redundant now
        fplist_all = []
        fp_manuvers = ["KS", "AC", "DC", "LCL", "LCR"]  # same order as in simulator
        ACCELERATION = (
            5  # TODO this value must fit the car and be matched in simulation
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
                map_lanes_nz[0, 0, closest_lane_id, :, 0].cpu().detach().numpy()
            )
            y_lane_current = (
                map_lanes_nz[0, 0, closest_lane_id, :, 1].cpu().detach().numpy()
            )
            yaw_lane_current = map_lanes_nz[0, 0, closest_lane_id, :, 2].cpu().detach().numpy()

            # tx, ty, tyaw, _ = cp.calc_bspline_course_2(
            #     x_lane_current,
            #     y_lane_current,
            #     100,
            #     0.5,
            # )

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

            fp_target_speeds = np.array(
                [
                    max(c_speed, 1),  # KS
                    c_speed + ACCELERATION,  # ACC
                    max(c_speed - ACCELERATION, 1),  # DC
                    max(c_speed, 1),  # LCL
                    max(c_speed, 1),  # LCR
                ]
            )

            # these params are the same for every maneuver now, since we attend different target lanes now, so a lot of the code can be simplified
            fp_lateral_dists = np.array([0, 0, 0, 0, 0])
            #fp_plan_time = np.ones(5) * 2.0
            #fp_plan_dt = np.ones(5) * 0.1
            
            fp_plan_time = np.ones(5) * (40 * 0.1) # initially was 2.0 replacing it with len_fut. The scale added to match the init value
            fp_plan_dt = np.ones(5) * (20 * 0.01) # initially was 2.0 replacing it with predictor_fps

            # select parameters for this maneuver
            fp_lateral_dists = fp_lateral_dists[i : i + 1]
            fp_plan_time = fp_plan_time[i : i + 1]
            fp_plan_dt = fp_plan_dt[i : i + 1]
            fp_target_speeds = fp_target_speeds[i : i + 1]
            fp_params = np.transpose(
                [fp_target_speeds, fp_lateral_dists, fp_plan_time, fp_plan_dt]
            )

            ## Calculate Frenet Paths w.r.t to the iterables
            fplist = fp_helper.calc_frenet_paths(
                c_speed, c_d, c_d_d, c_d_dd, s0, fp_params
            )

            ## Converts Paths from Frenet to Cartesian Coordinate System
            fplist = fp_helper.calc_global_paths(fplist, x_lane_current, y_lane_current, yaw_lane_current)
            fplist_all.append(fplist[0])
        fplist = fplist_all

        # # Compares all calculated paths with Obstacles Prediction Data
        # fplist = fp_helper.check_collision_dynamic(
        #     fplist,
        #     dynamic_ob=[],
        #     ego_x=ego_pos[0],
        #     ego_y=ego_pos[1],
        # )

        # ## Filter out paths based on curvature limit, obstacle collision
        # fplist = fp_helper.check_paths(fplist, (40 * 0.1), (20 * 0.01))

        batch_size = 5  # No. of Manuvers

        frenet_points = 20
        frenet_trajectories = torch.zeros(
            [batch_size, frenet_points, 17]
        )  # TODO catch case where frenet_points > trajectory length

        manuver_laneid = {}
        for i, (traj, manuver) in enumerate(zip(fplist, fp_manuvers)):
            frenet_trajectories[i, ..., [0, 1, 2, 3]] = torch.tensor(
                [traj.x, traj.y, traj.yaw, traj.s_d], 
            ).T[:frenet_points, :].float()

            # Mapping Manuer class to LaneID
            last_traj_point = [traj.x[-1], traj.y[-1]]
            dists = []
            for lane_id in range(map_lanes_nz.shape[2]):
                last_lane_points = (
                    map_lanes_nz[0, 0, lane_id, :, :2].cpu().detach().numpy()
                )

                dists.append(
                    np.min(np.linalg.norm(last_lane_points - last_traj_point, axis=1))
                )

            manuver_laneid[manuver] = np.argmin(np.array(dists))

        # maneuver_traj_history.append(frenet_trajectories.unsqueeze(0))
        # maneuver_traj_history = self.maneuver_traj_history[-frenet_points:]
            ## Compares all calculated paths with Obstacles Prediction Data

        # find minimum cost path
        min_cost = float("inf")
        best_path_idx = None
        for i,fp in enumerate(fplist):
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path_idx = i

        if torch.any(torch.isnan(frenet_trajectories)):
            print(frenet_trajectories)

        return best_path_idx, frenet_trajectories