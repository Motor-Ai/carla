import math
import numpy as np



def aerodynamic_force(v):
    # The aerodynamic drag is proportional to the square of the speed: Fa =
    # 1/2 \rho Cd A |v| v, where \rho is the density of air, Cd is the
    # shape-dependent aerodynamic drag coefficient, and A is the frontal area
    # of the car.
    ## https://www.researchgate.net/publication/330092375_Modeling_of_an_Electric_Vehicle_with_MATLABSimulink

    ##Drag Coeff Values
    # Car 0.3 to 0.4
    # Bus 0.6 to 0.7
    # Truck 0.8 to 1.5
    Cd = 0.35  # drag coefficient
    rho = 1.3  # density of air, kg/m^3

    ## Vehicle dimensions here - https://www.mercedes-benz.de/passengercars/mercedes-benz-cars/models/eqv/specifications/space.module.html
    A = 3.66  # car area, m^2
    Fa = 1 / 2 * rho * Cd * A * abs(v) * v

    return Fa


def tire_rolling_resistance(v):
    # A simple model of rolling friction is Fr = m g Cr sgn(v), where Cr is
    # the coefficient of rolling friction and sgn(v) is the sign of v (±1) or
    # zero if v = 0.
    m = 3500  # vehicle mass, kg
    g = 9.8  # gravitational constant, m/s^2

    # Tire Rolling Resistance Coefficient (Cr) depends on surface & tyres
    # Concrete or Asphalt -> 0.013
    # Small Gravel Ground -> 0.02
    # Macadamized Road -> 0.025
    # Soil Road -> 0.1 to 0.35
    Cr = 0.01  # coefficient of rolling friction

    if v > 0.0:
        sign_vel = 1.0
    elif v < 0.0:
        sign_vel = -1.0
    else:
        sign_vel = 0.0

    Fr = m * g * Cr * sign_vel

    return Fr


def hill_climbing_force(slope_theta):
    # Letting the slope of the road be \theta (theta), gravity gives the
    # force Fg = m g sin \theta.
    m = 3500  # vehicle mass, kg
    g = 9.8  # gravitational constant, m/s^2

    Fhc = m * g * math.sin(slope_theta) ## Fhc is positive for uphill, & negative for downhill

    return Fhc


# def acceleration_force(a):
#     # There are two types of acceleration forces:
#     # Linear acceleration & Angular acceleration forces

#     # Angular force is defines as
#     # I = get_moment_of_inertia() # Moment of Inertia of the Rotor
#     # G = 13 # Gear Ratio between Motor & Tyre
#     # ng = # Gear System Efficiency
#     # r = # Tyre Radius 

#     # Fwa = ((I*G**2)/(ng*r**2))*a

#     # From Electric Vehicle Modelling (Larminie & Lowry) Book Pg 191:
#     # Typical values for the constants here are 40 for G/r and 0.025 kg m2 for the moment
#     # of inertia. These are for a 30 kW motor, driving a car which reaches 60 kph at a motor
#     # speed of 7000 rpm. Such a car would probably weigh about 800 kg. The I G2/r2 term in
#     # above Equation will have a value of about 40 kg in this case. In other words, the angular
#     # acceleration force given by above Equation will typically be much smaller than the linear
#     # acceleration force given by Equation (F = ma).

#     #     It will quite often turn out that the moment of inertia of the motor I will not be known. In
#     # such cases a reasonable approximation is simply to increase the mass by 5% in Equation
#     # and to ignore the Fwa term.


#     # Linear Acceleration is define by
#     m = 3500 # vehicle mass, kg

#     Fla = 1.05*m*a ## Negative when slowing down (or decelerating)

#     return Fla


def tractive_force_speed_curve_EQV(throttle_perc, speed_mps):
    """
    Approximates the Force vs Speed Curve, during Acceleration of the EQV based on Shadow-Mode Data
    """
    mass = 3500 #kg
    speed_kph = speed_mps*3.6 # kph
    transition_kph = 45.0 # kph
    exp_curve_transition_acc = (np.exp(68.0/transition_kph)-1.0)*(throttle_perc/100.0)
    
    if speed_kph < transition_kph:
        return exp_curve_transition_acc * mass
    
    else:
        exp_curve_acc = (np.exp(68.0/speed_kph)-1.0)*(throttle_perc/100.0)
        return exp_curve_acc * mass


def acceleration_cmd_to_throttle_perc_EQV(Fd, mass, accel_cmd, ego_speed_mps):
    """
    Approximates the accel command to a Throttle value
    """
    mass = 3500 #kg
    ego_speed_kph = ego_speed_mps*3.6 # kph
    transition_kph = 45.0 # kph
    
    if ego_speed_kph < transition_kph:
        exp_curve_transition_acc = (np.exp(68.0/transition_kph)-1.0)
        throttle_perc = ((mass*accel_cmd + Fd)/mass) * (1/(exp_curve_transition_acc)) * 70.0
    else:
        exp_curve_acc = (np.exp(68.0/ego_speed_kph)-1.0)
        throttle_perc = ((mass*accel_cmd + Fd)/mass) * (1/(exp_curve_acc)) * 70.0
    return throttle_perc


def braking_force_speed_curve_EQV(brake_perc):
    """
    Approximates the Force vs Speed Curve, during Deceleration of the EQV based on Shadow-Mode Data
    """
    mass = 3500 #kg
    slope_acc_vs_full_brakes = 12.0
    
    decel = slope_acc_vs_full_brakes * (brake_perc/100.0)
    
    braking_force = decel*mass
    
    return braking_force


def vehicle_update(self, v, u, slope_theta):
    """Vehicle dynamics for cruise control system.

    Parameters
    ----------
    v : velocity 
         System state: car velocity in m/s
    u : float %
         System input: throttle, where throttle is
         a float between 0 and 1, gear is an integer between 1 and 5,
         and road_slope is in rad.

    Returns
    -------
    float
        Vehicle acceleration

    """
    mass = 3500 #kg
    # Define variables for vehicle state and inputs
    if u >= 0.0:
        # Force generated by the engine
        F = tractive_force_speed_curve_EQV(u, v) * 1.8 + 1300.0
    else:
        # Force generated by the brakes
        F = braking_force_speed_curve_EQV(u) * 20.0 + 1300.0

    # Disturbance forces
    #
    # The disturbance force Fd has three major components: Fg, the forces due
    # to gravity; Fr, the forces due to rolling friction; and Fa, the
    # aerodynamic drag.

    Fg = hill_climbing_force(slope_theta)

    Fr = tire_rolling_resistance(v)

    Fa = aerodynamic_force(v)

    # Final acceleration on the car
    Fd = (Fg + Fr + Fa) * 1.0
    F_long = F - Fd

    self.get_logger().info("F: " + str(round(F,1)))
    self.get_logger().info("Fl: " + str(round(F_long,1)))
    self.get_logger().info("Fd: " + str(round(Fd,1)))

    if v <= 0.0 and u < 0.0:
        dv = 0.0
    else:
        dv = (F - Fd) / mass

    return dv


def deceleration_cmd_to_braking_perc_EQV(Fd, mass, accel_cmd) :
    """
    Approximates the decel command to a Brake value
    """
    slope_acc_vs_full_brakes = 12.0
    
    braking_perc = 100.0 * (((mass * accel_cmd) + Fd)/(mass * slope_acc_vs_full_brakes))
    
    return braking_perc


def KS_steering_rate_cmd_2_steering_angle_cmd(steer_rate_cmd, ego_steer_angle, time_step):
    """
    Integrates the steer_rate command for one time step (for Simple Kinematic Car Model)
    """
    steer_angle_cmd = steer_rate_cmd * time_step + ego_steer_angle
    return steer_angle_cmd


def throttle_brake_switching_logic(self, accel_cmd, ego_vel):
    """
    Derives Logic to decide when throttling or braking is needed
    """
    self.get_logger().debug("Accel cmd before:   " + str(accel_cmd))
    vel_kph = ego_vel * 3.6
    #accel = np.clip((1.2/np.exp(vel_kph/14.0))-0.65, 0.0, -1.0)

    self.get_logger().debug("")
    #accel = np.clip((1.2/np.exp(vel_kph/14.0))-0.65, -1.0, 0.0)
    #accel = np.clip((0.55/np.exp(vel_kph/25.0))-0.45, -1.0, 0.0)
    #accel = (0.55/np.exp(vel_kph/25.0))-0.45
    accel = np.clip((0.8/np.exp(vel_kph/25.0))-0.45, -0.45, 0.0)

    self.get_logger().debug("Accel limit:   " + str(accel))
    #accel = min(-0.1,(1.2/np.exp(vel_kph/14.0))-0.65)
    
    if vel_kph < 12.0:
        tolerance = 0.0
    else:
        tolerance = 0.1 #m/s²

    throttle_brake = 'coast'

    if accel_cmd >= (accel + tolerance):
        throttle_brake = 'throttle'
    elif accel_cmd <= (accel - tolerance):
        throttle_brake = 'brake'
    
    return throttle_brake


def KS_acceleration_cmd_2_throttle_brake_cmd(self, accel_cmd, ego_vel, slope_theta):
    """
    Calculates throttle/brake % cmd based on the acceleration cmd & current ego velocity  (for Simple Kinematic Car Model)
    """
    mass = 3500 #kg


    # Disturbance forces
    #
    # The disturbance force Fd has three major components: Fg, the forces due
    # to gravity; Fr, the forces due to rolling friction; and Fa, the
    # aerodynamic drag.

    Fg = hill_climbing_force(slope_theta)

    Fr = tire_rolling_resistance(ego_vel)

    Fa = aerodynamic_force(ego_vel)

    disturbance_gain = 1.0 # Tuned
    Fd = (Fg + Fr + Fa) * disturbance_gain


    throttle_brake_logic = throttle_brake_switching_logic(self, accel_cmd, ego_vel)

    self.get_logger().debug("MPC SWITCHING LOGIC:   " + throttle_brake_logic)
    self.get_logger().debug("")
    self.get_logger().debug("")


    if throttle_brake_logic == "throttle":
        # Calculate Throttle % based on the acceleration command
        u_zero_accel = 78.5/(0.88 + np.exp(-0.065*ego_vel*3.6)) - 42.0
        u = np.clip(acceleration_cmd_to_throttle_perc_EQV(Fd, mass, accel_cmd, ego_vel) + u_zero_accel, 0.0, 100.0)

    elif throttle_brake_logic == "brake":
        # Calculate Brake % based on the deceleration command
        #switch_logic_level = (1.2/np.exp(ego_vel*3.6/14.0))
        #switch_logic_tolerance = 0.1
        accel_cmd_level_corrected = accel_cmd #+ switch_logic_level #+ switch_logic_tolerance

        self.get_logger().debug("Braking inputs: Fd " + str(round(Fd,1)) + "  acc: " + str(round(accel_cmd_level_corrected,1)))
        u = -1.0 * np.clip(-(deceleration_cmd_to_braking_perc_EQV(Fd, mass, accel_cmd_level_corrected)), 0.0, 30.0)

    elif throttle_brake_logic == "coast":
        # Zero inputs
        u = 0.0

    self.get_logger().debug("ENGINE COMMAND " + str(round(u,2)) + " %")
    self.get_logger().debug("")


    return u/100
