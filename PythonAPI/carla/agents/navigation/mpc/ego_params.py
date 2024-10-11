from math import radians


class SteeringParameters:
    def __init__(self):
        # constraints regarding steering
        self.wheel_min = []  # minimum steering wheel angle [rad]
        self.wheel_max = []  # maximum steering wheel angle [rad]
        self.min = []  # minimum steering angle [rad]
        self.max = []  # maximum steering angle [rad]
        self.v_min = []  # minimum steering velocity [rad/s]
        self.v_max = []  # maximum steering velocity [rad/s]


class LongitudinalParameters:
    def __init__(self):
        # constraints regarding longitudinal dynamics
        self.v_min = None  # minimum velocity [m/s]
        self.v_max = None  # minimum velocity [m/s]
        self.v_switch = None  # switching velocity [m/s]
        self.a_max = None  # maximum absolute acceleration [m/s^2]


class VehicleParameters:
    def __init__(self):
        # vehicle body dimensions
        self.l = None
        self.w = None

        # steering parameters
        self.steering = SteeringParameters()

        # longitudinal parameters
        self.longitudinal = LongitudinalParameters()

        # mass
        self.m = None

        # axes distances
        self.a = None  # distance from spring mass center of gravity to front axle [m]
        self.b = None  # distance from spring mass center of gravity to rear axle [m]

        # moments of inertia of sprung mass
        self.I_z = None  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ

        # suspension parameters
        self.K_sf = None  # suspension spring rate (front) [N/m]  KSF
        self.K_sdf = None  # suspension damping rate (front) [N s/m]  KSDF
        self.K_sr = None  # suspension spring rate (rear) [N/m]  KSR
        self.K_sdr = None  # suspension damping rate (rear) [N s/m]  KSDR

        # geometric parameters
        self.T_f = None  # track width front [m]  TRWF
        self.T_r = None  # track width rear [m]  TRWB

        # center of gravity
        self.h_s = None  # M_s center of gravity above ground [m]  HS

        # tyre parameters
        self.R_w = None  # effective wheel/tyre radius  chosen as tyre rolling radius RR  taken from ADAMS documentation [m]
        self.tyre_p_dy1 = None  # Lateral friction Muy
        self.tyre_p_ky1 = None  # Maximum value of stiffness Kfy/Fznom

        self.path_res = None  # Lane resolution


def acceleration_constraints(velocity, acceleration, p):
    # accelerationConstraints - adjusts the acceleration based on acceleration 
    # constraints
    #
    # Syntax:  
    #    accelerationConstraints(velocity,acceleration,p)
    #
    # Inputs:
    #    acceleration - acceleration in driving direction
    #    velocity - velocity in driving direction
    #    p - longitudinal parameter structure
    #
    # Outputs:
    #    acceleration - acceleration in driving direction
    
    # positive acceleration limit
    if velocity > p.v_switch:
        posLimit = p.a_max * p.v_switch / velocity
    else:
        posLimit = p.a_max

    # acceleration limit reached?
    if (velocity <= p.v_min and acceleration <= 0) or (velocity >= p.v_max and acceleration >= 0):
        acceleration = 0
    elif acceleration <= -p.a_max:
        acceleration = -p.a_max
    elif acceleration >= posLimit:
        acceleration = posLimit

    return acceleration


def steering_constraints(steering_angle, steering_velocity, p):
    # steeringConstraints - adjusts the steering velocity based on steering
    # constraints
    #
    # Syntax:  
    #    steeringConstraints(steering_angle,steering_velocity,p)
    #
    # Inputs:
    #    steering_angle - steering angle
    #    steering_velocity - steering velocity
    #    p - steering parameter structure
    #
    # Outputs:
    #    steering_velocity - steering velocity

    # steering limit reached?
    if (steering_angle <= p.min and steering_velocity <= 0) or (steering_angle >= p.max and steering_velocity >= 0):
        steering_velocity = 0
    elif steering_velocity <= p.v_min:
        steering_velocity = p.v_min
    elif steering_velocity >= p.v_max:
        steering_velocity = p.v_max

    return steering_velocity
    

def parameters_EQV():
    """
    parameters_EQV - parameter set of the single-track vehicle dynamics
    based on the DOT (department of transportation) vehicle dynamics
    values are taken from a  - Mercedes Benz EQV 300 Long 2021
    """

    # init vehicle parameters
    p = VehicleParameters()

    # vehicle body dimensions
    p.l = 5.140  # vehicle length [m]
    p.w = 2.244  # vehicle width [m]

    # steering constraints
    p.steering.wheel_min = -radians(580.0)  # minimum steering wheel angle [rad]
    p.steering.wheel_max = radians(580.0)  # minimum steering wheel angle [rad]
    p.steering.min = -radians(32.75)  # minimum steering angle [rad]
    p.steering.max = radians(32.75)  # maximum steering angle [rad]

    # p.steering.v_min = -radians(2.5)  #minimum steering velocity [rad/s] # Also Varies based on a speed_vs_steer_rate curve
    # p.steering.v_max = radians(2.5)  #maximum steering velocity [rad/s] # Also Varies based on a speed_vs_steer_rate curve
    # p.steering.v_min = -radians(37.75)  #minimum steering velocity [rad/s] # Also Varies based on a speed_vs_steer_rate curve
    # p.steering.v_max = radians(37.75)  #maximum steering velocity [rad/s] # Also Varies based on a speed_vs_steer_rate curve
    p.steering.v_min = -radians(
        12.0
    )  # minimum steering velocity [rad/s] # Also Varies based on a speed_vs_steer_rate curve
    p.steering.v_max = radians(
        12.0
    )  # maximum steering velocity [rad/s] # Also Varies based on a speed_vs_steer_rate curve

    # longitudinal constraints
    p.longitudinal.v_min = -10.0  # minimum velocity [m/s]
    p.longitudinal.v_max = 44.44  # maximum velocity [m/s]
    p.longitudinal.v_switch = 7.824  # switching velocity [m/s]
    p.longitudinal.a_max = 11.5  # maximum absolute acceleration [m/s^2]

    # mass
    p.m = 3500.0  # vehicle mass [kg]  MASS

    # axes distances
    p.a = 1.65  # distance from spring mass center of gravity to front axle [m]
    p.b = 1.55  # distance from spring mass center of gravity to rear axle [m]

    # moments of inertia of sprung mass
    p.I_z = 5800.0  # moment of inertia for sprung mass in yaw [kg m^2] - Izz

    # geometric parameters
    p.T_f = 1.68  # track width front [m]
    p.T_r = 1.65  # track width rear [m]

    # center of gravity
    p.h_s = 0.6  # M_s center of gravity above ground [m]  HS

    # tyre parameters
    p.R_w = 0.32  # effective wheel/tyre radius  chosen as tyre rolling radius RR  taken from ADAMS documentation [m]
    ##tyre lateral coefficients
    p.tyre_p_dy1 = 1.0489  # Lateral friction Muy
    p.tyre_p_ky1 = -21.92  # Maximum value of stiffness Kfy/Fznom

    return p