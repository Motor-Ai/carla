from math import atan2, asin, pi
from numpy import clip, cos, sin, tan, arctan2, radians, array, hypot, argmin, dot, mean, degrees, exp, zeros
from agents.navigation.mpc import ego_params as ep
import matplotlib.pyplot as plt
import math
import numpy as np
import acado

NX = 5  # [x, y, v, yaw, delta]
NY = 6
NYN = 4
NU = 2  # [accel, deltarate]
T = 40  # horizon length

# Q = np.diag([x,  y,   v,   phi,  a,   deltarate])
Q  = np.diag([0.05, 0.05, 0.8, 0.1, 2.0, 5.0])  # state cost matrix
Qf = np.diag([0.05, 0.05, 0.8, 0.1])  # state cost matrix

DU_TH = 0.1  # iteration finish param
DT = 0.05  # [s] time tick

MAX_STEER = np.deg2rad(30.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(90.0/4.5)  # maximum steering speed [rad/s]
MAX_ACCEL = 5.0  # maximum accel [m/ss]
MAX_DECEL = 3.0  # maximum decel [m/ss]
MAX_SPEED = 60.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = 0.1 / 3.6  # minimum speed [m/s]

N_IND_SEARCH = 200  # Search index number


################# Temporary - Update with current parameters ###################
# Vehicle parameters
WB = 3.2  # [m]



class State(object):
    """
    Class representing the state of a vehicle.
    :param x: (float) x-coordinate [m]
    :param y: (float) y-coordinate [m]
    :param yaw: (float) yaw angle [rad]
    :param v: (float) speed [m/s]
    :param dt: (float) time per frame [s]
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, dt=0.05, delta=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.dt = dt
        self.delta = delta



def pi_2_pi(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle



def calc_nearest_index(state, cx, cy, cyaw):
    """
    Calculate nearest point on path w.r.t state.
    :param state: (object) ego-state Object
    :param cx: (float arr)  reference path x array [m]
    :param cy: (float arr)  reference path y array [m]
    :param cyaw: (float arr)  reference path yaw array [rad]
    :return: (int)
    """

    # Calc front axle position
    fx = state.x # + WB * math.cos(state.yaw) * 0.5
    fy = state.y # + WB * math.sin(state.yaw) * 0.5

    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    mind = math.sqrt(mind)

    dxl = cx[ind] - fx
    dyl = cy[ind] - fy

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind



def find_front_axle_idx(state, path_x, path_y, delay):
    """
    Calculate nearest point on path w.r.t state.
    :param state: (object) ego-state Object
    :param path_x: (float arr)  reference path x array [m]
    :param path_y: (float arr)  reference path y array [m]
    :return: (int)
    """

    # Reaction Delay
    prediction_dist = state.v * delay

    # Calc front axle position
    #fx = state.x + WB * math.cos(state.yaw)
    #fy = state.y + WB * math.sin(state.yaw)

    fx = state.x + (WB + prediction_dist) * math.cos(state.yaw)
    fy = state.y + (WB + prediction_dist) * math.sin(state.yaw)

    dx = [fx - icx for icx in path_x]
    dy = [fy - icy for icy in path_y]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    front_axle_vec = [-np.cos(state.yaw + np.pi / 2), -np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[ind], dy[ind]], front_axle_vec)

    return ind, error_front_axle, [fx,fy]


def delay_corrected_state(state, delay, min_vel, max_vel):
    """
    State corrected based on actuator delay
    """
    new_state = state
    vel = np.clip(state.v, min_vel, max_vel)

    new_state.x = state.x + vel * math.cos(state.yaw) * delay
    new_state.y = state.y + vel * math.sin(state.yaw) * delay
    new_state.yaw = state.yaw + vel / WB * math.tan(state.delta) * delay

    return new_state



def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl):
    """
    Calculate nearest point on path w.r.t state.
    :param state: (object) ego-state Object
    :param cx: (float arr)  reference path x array [m]
    :param cy: (float arr)  reference path y array [m]
    :param cyaw: (float arr)  reference path yaw array [rad]
    :param ck: (flaot arr)  reference path curvature [rad/s]
    :param sp: (float arr)  reference path speed profile [m/s]
    :param dl: (int)  reference path resolution 0.1 m
    return: (float arr, int, float arr)
    """
    xref = np.zeros((NY, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind = calc_nearest_index(state, cx, cy, cyaw)

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    #xref[2, 0] = state.v
    xref[3, 0] = cyaw[ind]
    xref[4, 0] = 0.0
    xref[5, 0] = 0.0
    dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            #xref[2, i] = state.v
            xref[3, i] = cyaw[ind + dind]
            xref[4, i] = 0.0
            xref[5, i] = 0.0
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            #xref[2, i] = state.v
            xref[3, i] = cyaw[ncourse - 1]
            xref[4, i] = 0.0
            xref[5, i] = 0.0
            dref[0, i] = 0.0

    return xref, ind, dref



def iterative_linear_mpc_control(self, xref, x0, dref, oa, od, ego_vel, lane_morph, target_speed, throttle_value_change):
    """
    MPC control with updating operational point iteraitvely
    :param xref: (float arr) reference trajectory points
    :param x0: (float arr) current state
    :param dref: (zeros arr) reference steerings
    :param oa: previous iteration acceleration output used in predict motion 
    :param od: previous iteration steering output used in predict motion
    :return: (float arr, float arr, float arr, float arr, float arr, float arr)
    """
    weight = 0.0
    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for i in range(T):                                       ## MAX_ITER can be replaced by T
        self.get_logger().debug('X0:  ' + str(np.round(np.degrees(x0[4]))))
        self.get_logger().debug('Yaw_ref' + str(np.round(np.degrees(xref[3]))))

        xbar = predict_motion(self, x0, oa, od, xref)

        self.get_logger().debug('Yaw_pred' + str(np.round(np.degrees(xbar[3]))))
        xref_corr = xref
        if abs(xref_corr[3][0] - xbar[3][0]) > np.radians(90.0):
            if xbar[3][0] > 0.0:
                xref_corr[3][0] = xref_corr[3][0] + np.radians(360.0)
            elif xbar[3][0] < 0.0:
                xref_corr[3][0] = xref_corr[3][0] - np.radians(360.0)
            xref_corr[3] = smooth_yaw(xref_corr[3])
            self.get_logger().debug('Yaw_ref corr' + str(np.round(np.degrees(xref_corr[3]))))

        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov, weight = linear_mpc_control(self, xref_corr, xbar, x0, dref, ego_vel, lane_morph, target_speed, throttle_value_change)

        self.get_logger().debug('ODelta' + str(np.round(np.degrees(od))))

        #du = sum(abs(od - pod))   # calc u change value
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value

        self.get_logger().debug("")
        self.get_logger().debug("")

        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov, weight



# MPC using ACADO
def linear_mpc_control(self, xref, xbar, x0, dref, ego_vel, lane_morph, target_speed, throttle_value_change):
    """
    Call ACADO package and execute MPC calculations
    :param xref: (float arr) reference trajectory points
    :param xbar: (float arr) prediction trajectory points
    :param x0: (float arr) current state
    :param dref: (float arr) steering reference
    :return: (float arr, float arr, float arr, float arr, float arr, float arr)
    """
    # see acado.c for parameter details
    # x0: The initial state of the system
    # X : The predicted states of the system for the next (T) prediction time steps using the last control input (u)
    # Y : The reference states of the system for the next (T) prediction time steps
    # yN: The final reference state of the system
    # U : The returned (optimal) control input that controls the process into the reference states
    # Q : A reference states cost matrix for the the next (T) prediction time steps
    # Qf: A reference state cost matrix for the final state

    Q  = np.diag([0.05, 0.05, 1.0, 1.0, 1.0, 5.0])  # state cost matrix
    Qf = np.diag([0.05, 0.05, 1.0, 1.0])  # state cost matrix

    if lane_morph in ['turn', 'u-turn', 'high_curvature']:
        Q_steer_rate_weight_pow = 2.0
        Qx_p = 0.01
        Qy_p = 0.01
        Qyaw_p = 0.5

    elif lane_morph in ['roundabout']:
        Q_steer_rate_weight_pow = 2.0
        Qx_p = 0.01
        Qy_p = 0.01
        Qyaw_p = 1.0
    
    elif lane_morph in ['high_speed']:
        if ego_vel < (20.0 / 3.6):
            if target_speed > ego_vel:
                Q_steer_rate_weight_pow = 2.0
            else:
                Q_steer_rate_weight_pow = 15.0
        else:
            if target_speed > ego_vel:
                Q_steer_rate_weight_pow = 5.0
            else:
                Q_steer_rate_weight_pow = 10.0

        Qx_p = 0.03
        Qy_p = 0.03
        Qyaw_p = 1.0
        self.get_logger().debug('highspeed ' + str(np.round(target_speed-ego_vel,2)))

    elif lane_morph in ['normal']:
        Q_steer_rate_weight_pow = 5.0
        self.get_logger().debug('normal ' + str(np.round(target_speed-ego_vel,2)))
        Qx_p = 0.04
        Qy_p = 0.04
        Qyaw_p = 1.0

    else:
        Q_steer_rate_weight_pow = 2.0
        Qx_p = 0.02
        Qy_p = 0.02
        Qyaw_p = 1.0

    #self.Qxy.append(Qx_p)
    #self.Qyaw.append(Qyaw_p)
    #self.Qsteer.append(Q_steer_rate_weight_pow)

    Q[5,5] = Q_steer_rate_weight_pow
    Q[0,0] = Qx_p
    Q[1,1] = Qy_p
    Qf[0,0] = Qx_p
    Qf[1,1] = Qy_p
    Q[3,3] = Qyaw_p
    Qf[3,3] = Qyaw_p

    # Q[5,5] = np.mean(self.Qsteer[-90:])
    # Q[0,0] = np.mean(self.Qxy[-90:])
    # Q[1,1] = np.mean(self.Qxy[-90:])
    # Qf[0,0] = np.mean(self.Qxy[-90:])
    # Qf[1,1] = np.mean(self.Qxy[-90:])
    # Q[3,3] = np.mean(self.Qyaw[-90:])
    # Qf[3,3] = np.mean(self.Qyaw[-90:])

    #self.Qxy = self.Qxy[-100:]
    #self.Qyaw = self.Qyaw[-100:]
    #self.Qsteer = self.Qsteer[-100:]

    #Q_steer_rate_weight_pow = np.round(np.clip((ego_vel*3.6)*0.16, 1,8),1)
    #Q[5,5] = Q_steer_rate_weight_pow



    _x0=np.zeros((1,NX))  
    X=np.zeros((T+1,NX))
    U=np.zeros((T,NU))    
    Y=np.zeros((T,NY))    
    yN=np.zeros((1,NYN))    
    _x0[0,:]=np.transpose(x0)  # initial state    
    for t in range(T):
      Y[t,:] = np.transpose(xref[:,t])  # reference state
      X[t,:] = np.transpose(xbar[:,t])  # predicted state
    X[-1,:] = X[-2,:]    
    yN[0,:]=Y[-1,:NYN]         # reference terminal state
    X, U = acado.mpc(0, 1, _x0, X,U,Y,yN, np.transpose(np.tile(Q,T)), Qf, 0)    
    ox = get_nparray_from_matrix(X[:,0])
    oy = get_nparray_from_matrix(X[:,1])
    ov = get_nparray_from_matrix(X[:,2])
    oyaw = get_nparray_from_matrix(X[:,3])
    oa = get_nparray_from_matrix(U[:,0])
    odelta = get_nparray_from_matrix(U[:,1])
    return oa, odelta, ox, oy, oyaw, ov, Q_steer_rate_weight_pow



def get_nparray_from_matrix(x):
    """
    post processing on array after ACADO operation
    :param x: (float arr)
    :return: (flatten arr)
    """
    return np.array(x).flatten()



def predict_motion(self, x0, oa, od, xref):
    """
    predict motion with current state to calculate differential variable vectors for acado.
    :param x0: (float arr) current state
    :param oa: (float oa) previous iteration acceleration output used in predict motion 
    :param od: (float od) previous iteration steering output used in predict motion
    :param xref: (float arr) reference trajectory points
    :return: (float arr)
    """
    #xbar = xref * 0.0
       
    xbar = np.zeros((NX, T + 1))    
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2], dt=x0[4])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):       
        state = update_state(state, ai, di, DT, True)        
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw
        xbar[4, i] = state.delta

    #self.get_logger().debug('ODelta' + str(np.round(np.degrees(od))))
    #self.get_logger().debug('Yaw_pred' + str(np.round(np.degrees(xbar[3]))))
    #self.get_logger().debug('Yaw_ref' + str(np.round(np.degrees(xref[3]))))
    #xbar[3] = normalize_yaw_pred(xref[3], xbar[3])
    #yaw_pred = normalize_yaw_pred(xref[3], xbar[3])
    #self.get_logger().debug('Yaw_pred 2' + str(np.round(np.degrees(yaw_pred))))

    return xbar



def normalize_yaw_pred(yaw_ref, yaw_pred):
    yaw_pred_normalized = []

    flag = np.all(np.abs(yaw_ref - yaw_pred) >= np.pi/2.0)
    #flag_2 = 

    if flag:
        for i in range(len(yaw_pred)):
            yaw_diff = yaw_ref[i] - yaw_pred[i]
            yaw_diff_normalized = pi_2_pi(yaw_diff)
            yaw_pred_normalized.append(yaw_ref[i] - yaw_diff_normalized)
    else:
        yaw_pred_normalized = yaw_pred
    return np.array(yaw_pred_normalized)



def update_state(state, a, deltarate, dt, predict):
    """
    update state of vehicle in predict motion.
    :param state: (object) ego-state Object
    :param ai: (float arr) acceleration input
    :param di: (float arr) steering rate input
    :param DT: (float) 0.05
    :param predict: (bool)
    """

    # input check
    if deltarate > MAX_DSTEER:
        deltarate = MAX_DSTEER
    elif deltarate < -MAX_DSTEER:
        deltarate = -MAX_DSTEER        
       
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / WB * math.tan(state.delta) * dt
    state.v = state.v + a * dt
    state.delta = state.delta + deltarate * dt
    
    if state.delta > MAX_STEER:
        state.delta = MAX_STEER
    elif state.delta < -MAX_STEER:
        state.delta = -MAX_STEER


    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED    

    return state



def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > pi:
        angle -= 2.0 * pi

    while angle < -pi:
        angle += 2.0 * pi

    return angle



def smooth_yaw(yaw):
    """
    Removes the sign changes that occur in the input array
    to ensure continuous change in the yaw angle
    :param yaw: (float arr)
    :return: (float arr)
    """
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= pi / 2.0:
            yaw[i + 1] -= pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -pi / 2.0:
            yaw[i + 1] += pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw



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
    roll_x = atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians



def plot_arrow(x, y, yaw, length=1.0, width=0.6, fc="r", ec="k"):
    """
    Plots arrow
    :param x: (float) ego x
    :param y: (float) ego y
    :param yaw: (float) ego yaw
    """
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


def get_car_outline(x, y, yaw):
    """
    Returns the Polygon of the Ego Vehicle based on the Dimensions, Given Input Ego Pose
    :param x: (float) ego x
    :param y: (float) ego y
    :param yaw: (float) ego yaw
    """
    # EQV/EVito Dimensions
    LENGTH = 5.140  # [m]
    WIDTH = 2.23  # [m]
    BACKTOWHEEL = 1.045  # [m] rear axle center to the back of the car
    outline = array(
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

    Rot1 = array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])

    outline = (outline.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y

    return outline.T


def constraint_di(di, last_di, dt, steer_ratio, max_steer_rate):
    """
    Adjust rate of chage of steering val
    :param di: (float)
    :param last_di: (float)
    :param dt: (float)
    :param steer_ratio: (float)
    :return: (float)
    """

    ## Setting Rate of Return of Steering to 0.0 degs
    max_di_change = radians(max_steer_rate) * steer_ratio

    di_diff = di - last_di

    if di_diff >= 0.0:
        if abs(di_diff) > max_di_change * dt:
            new_di = last_di + max_di_change * dt
        else:
            new_di = di

    elif di_diff < 0.0:
        if abs(di_diff) > max_di_change * dt:
            new_di = last_di - max_di_change * dt
        else:
            new_di = di

    return new_di


def steer_err_based_target_speed(steer_err):
    '''
    Calculates the safe_speed w.r.t to the steer_err input
    :param steer_err: (float) [rad]
    :return: (float) [m/s]
    '''
    steer_err_gain = 0.4
    min_target_speed = 8.5 #kph
    kph_2_mps_ratio = 3.6

    eq_num = 100.0
    exp_diff = 0.7
    exp_div = 1.3
    return ((eq_num/(exp((degrees(abs(steer_err*steer_err_gain)) - exp_diff)/exp_div))) + min_target_speed)/kph_2_mps_ratio


def lat_error_based_target_speed(lat_error):
    '''
    Calculates the safe_speed w.r.t to the steer_err input
    :param steer_err: (float) [rad]
    :return: (float) [m/s]
    '''
    lat_error_gain = 0.4
    min_target_speed = 8.5 #kph
    kph_2_mps_ratio = 3.6

    eq_num = 15.0
    exp_diff = 0.7
    exp_div = 1.3
    return ((eq_num/(np.exp(abs(lat_error*lat_error_gain) - exp_diff)/exp_div)) + min_target_speed)/kph_2_mps_ratio