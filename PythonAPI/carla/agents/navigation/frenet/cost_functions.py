import numpy as np
import torch
import torch.nn.functional as F
import stlcg
from stlcg import Expression

# from dipp_predictor_py.stlcg_predicates import stop_sign_rule

cost_weights = [1, 0.5, 0.1, 0.1, 1, 0, 10]
SPEED_LIMIT = 5  # m/s # TODO extract from map
MIN_SAFE_DIST = 10  # m


def calculate_cost(predictions, map_lanes):
    route_goal = (
        map_lanes[:, 0, 0, -1, :2].unsqueeze(1).repeat(1, predictions.shape[1], 1)
    )  # [5, 5, 2]

    ego_pred = predictions[:, :, 0]
    ego_pred_pos = ego_pred[:, :, -1, :2]  # [5,5,2]
    neigh_pred = predictions[:, :, 1:]

    route = map_lanes[0, 0, 0, : ego_pred.shape[-2], :2]
    route = (
        route.unsqueeze(0)
        .unsqueeze(0)
        .repeat(predictions.shape[0], predictions.shape[1], 1, 1)
    )

    speed = check_speed(ego_pred) / SPEED_LIMIT * cost_weights[1]

    d2g = min_dist(ego_pred, route)

    lon_jerk, lat_acc = check_comfort(ego_pred)
    lon_jerk, lat_acc = lon_jerk * cost_weights[2], lat_acc * cost_weights[3]

    collision, ttc, d2a = check_collision(ego_pred, neigh_pred)
    collision, ttc, d2a = (
        collision * cost_weights[6],
        ttc * cost_weights[5],
        d2a * cost_weights[4],
    )

    # STL robustness
    speed_stl_robustness = stl_robustness(speed, threshold_value=5)
    d2g_stl_robustness = stl_robustness(d2g, threshold_value=5)
    lon_jerk_stl_robustness = stl_robustness(lon_jerk, threshold_value=5)
    lat_acc_stl_robustness = stl_robustness(lat_acc, threshold_value=5)
    d2a_stl_robustness = stl_robustness(d2a, threshold_value=5)
    ttc_stl_robustness = stl_robustness(ttc, threshold_value=5)

    cost = torch.sum(
        torch.stack([d2g, speed, lon_jerk, lat_acc, d2a, ttc, collision]), dim=0
    )

    cost_names = ["d2g", "speed", "lon_jerk", "lat_acc", "d2a", "ttc", "collision"]
    return cost, [d2g, speed, lon_jerk, lat_acc, d2a, ttc, collision], cost_names


def stl_robustness(cost_val: torch.Tensor, threshold_value):
    cost_val = cost_val.view(cost_val.shape[0] * cost_val.shape[1], -1).unsqueeze(-1)
    signal = Expression("cost_signal", torch.randn((1)).requires_grad_(False))
    criteria = stlcg.LessThan(lhs=signal, val=threshold_value)
    return criteria.robustness(cost_val)


def min_dist(ego_pred, route_clipped):
    route_to_goal_dist = torch.norm(
        route_clipped[..., :2] - ego_pred[..., :1, :2], dim=-1
    )

    # Clip the route to ego pred length # TODO: Make the time space relative
    route_to_goal_dist = route_to_goal_dist[:, : ego_pred.shape[-2]]

    # route_to_goal_dist = torch.norm(route_clipped[..., :2] - ego_pred, dim=-1)

    # Considering ego cnetric route
    # route_to_goal_dist = torch.norm(route_clipped[:, :2], dim=-1)
    # dist_to_goal, idx = route_to_goal_dist.min(dim=-1)
    return route_to_goal_dist


# STL rule implementation
def check_speed_stl(ego_pred):
    # Speed must be less than the set threshold, but also close to the set threshold
    lon_speed = torch.diff(ego_pred[..., 0], dim=-1)
    speed_upper_limit = Expression("speed_upper_limit", SPEED_LIMIT)
    speed_lower_limit = Expression("speed_lower_limit", SPEED_LIMIT - 5)
    phi1 = stlcg.LessThan("signal", speed_upper_limit)
    phi2 = stlcg.GreaterThan("signal", speed_lower_limit)
    phi = stlcg.Always(subformula=phi1 & phi2)
    lon_speed = lon_speed.swapaxes(-1, -2)
    speed_robustness = phi.robustness((lon_speed, lon_speed))
    speed_robustness = speed_robustness.swapaxes(-1, -2).squeeze(-1)
    return torch.relu(-speed_robustness)


def check_stop_line_sign(ego_pred, route):
    ss = stop_sign_rule(ego_pred, route)
    ss_robustness = ss.robustness(ego_pred)
    return torch.relu(-ss_robustness)


def check_speed(ego_pred):
    # acc = traj[:, 0]
    # dt = 0.01
    lon_speed = torch.diff(ego_pred[..., 0], axis=-1)
    speed_diff = torch.abs(lon_speed - SPEED_LIMIT)
    # speed_diff = torch.mean(speed_diff, dim=-1)
    speed_diff = F.pad(speed_diff, (0, 1), "constant", 0)
    return speed_diff


def check_dist_to_goal(route, ego_pose):
    route = route[-1, :2].unsqueeze(0).repeat((ego_pose.shape[0], 1))
    dist = torch.norm(ego_pose[:, 0, -1, :2] - route, dim=-1)
    return dist


def check_collision(ego_pred, neigh_pred):
    # plan_frenet = np.column_stack([ego_pred.s, ego_pred.d])
    # ego_pred = np.column_stack([ego_pred.x, ego_pred.y])

    # Get ego_pred from predictions
    collision = 0
    dist = [100]

    # Torch implementation
    neigh_pred[neigh_pred[..., 0] <= 1.0] = 500
    neigh_pred[neigh_pred[..., 1] <= 1.0] = 500
    ego_pred = ego_pred.unsqueeze(dim=-3).repeat(1, 1, neigh_pred.shape[2], 1, 1)

    dist = torch.norm(ego_pred[..., :2] - neigh_pred[..., :2], dim=-1)

    # calcualte collision and TTC if any obstacle below min distance
    try:
        # collision = (dist < MIN_SAFE_DIST).any(dim=-1).any(dim=-1).type(torch.uint8)
        ttc = (dist < MIN_SAFE_DIST).type(torch.uint8)

        # # TODO: Improve the following
        # ttc = torch.ones_like(collision) * 3
        # for batch in range(ttc.shape[0]):
        #     for traj in range(ttc.shape[1]):
        #         for veh in range(ttc.shape[2]):
        #             id = torch.where(dist[batch, traj, veh] < MIN_SAFE_DIST)[0]
        #             if (
        #                 torch.where(dist[batch, traj, veh] < MIN_SAFE_DIST)[0].shape[0]
        #                 != 0
        #             ):
        #                 ttc[batch, traj, veh] = id[-1]

        # ttc = (ttc + 1) / 10
    except:
        pass

    # Find the min dist across ma and samples
    min_dist = torch.amin(dist, dim=-2)  # TODO:dim = -1 or -2?
    min_dist = torch.clip(min_dist - 3, 0, 20)
    d2a = torch.exp(-0.3 * min_dist**2)
    ttc = 1 - ttc / 3
    ttc, _ = torch.min(ttc, dim=-2)
    collision = (dist < MIN_SAFE_DIST).any(dim=-2).type(torch.float16)
    return collision, ttc, d2a


def check_comfort(ego_pred):
    v_x, v_y, theta = (
        torch.diff(ego_pred[..., 0]) / 0.1,
        torch.diff(ego_pred[..., 1]) / 0.1,
        ego_pred[..., 2],
    )
    v_x = torch.nn.functional.pad(v_x, (1, 0), value=0)  # Diff results in reduced shape
    v_y = torch.nn.functional.pad(v_y, (1, 0), value=0)  # Diff results in reduced shape
    lon_speed = v_x * torch.cos(theta) + v_y * torch.sin(theta)
    lat_speed = v_y * torch.cos(theta) - v_x * torch.sin(theta)
    acc = torch.diff(lon_speed, n=1) / 0.1
    jerk = torch.diff(acc, n=1)
    lat_acc = torch.diff(lat_speed)

    jerk = torch.nn.functional.pad(
        jerk, (2, 0), value=0
    )  # Diff results in reduced shape
    lat_acc = torch.nn.functional.pad(
        lat_acc, (1, 0), value=0
    )  # Diff results in reduced shape
    # jerk = torch.mean(torch.abs(jerk), dim=-1) / 10
    # lat_acc = torch.mean(torch.abs(lat_acc), dim=-1) / 10

    return jerk, lat_acc


if __name__ == "__main__":
    predictions = torch.randn([1, 5, 11, 30, 3])
    map_lanes = torch.randn([1, 1, 2, 1000, 17])
    calculate_cost(predictions, map_lanes)
