from numpy import sign, clip


def anti_windup_protection(target, current, throttle_gas_level, ai, I_term):
    """
    Windup Protection
    :param target: (float)
    :param current: (float)
    :param throttle_gas_level: (float)
    :param ai: (float)
    :param I_term: (float)
    :return: (float)
    
    """

    # Condition 1
    if sign(target - current) == sign(throttle_gas_level):
        bool_1 = True
    else:
        bool_1 = False

    # Condition 2
    if throttle_gas_level != ai:
        bool_2 = True
    else:
        bool_2 = False

    # if both conditions are true then zero the I_term
    if bool_1 and bool_2:
        I_term = 0.0

    return I_term


def constraint_throttle(
    throttle,
    last_throttle,
    dt,
    max_throttle_change_accel,
    max_throttle_change_accel_release,
    max_throttle_change_brake,
    max_throttle_change_brake_release,
):
    """
    Constraints the throttle & braking rates
    :param throttle: (float)
    :param last_throttle: (float)
    :param dt: (float)
    :return: (float)
    """

    throttle_diff = throttle - last_throttle

    if throttle_diff >= 0.0:
        if last_throttle <= 0.0 and throttle >= 0.0:
            brake_release_diff = abs(last_throttle)
            brake_release_change = clip(
                brake_release_diff, 0.0, max_throttle_change_brake_release * dt
            )

            if brake_release_diff >= max_throttle_change_brake_release:
                accel_press_diff = 0.0
                accel_press_change = 0.0
            else:
                accel_press_diff = (
                    max_throttle_change_brake_release - brake_release_change
                )
                accel_press_change = clip(
                    accel_press_diff, 0.0, max_throttle_change_accel * dt
                )

            final_throttle = last_throttle + brake_release_change + accel_press_change

        elif last_throttle >= 0.0 and throttle >= 0.0:
            accel_press_diff = abs(throttle_diff)
            accel_press_change = clip(
                accel_press_diff, 0.0, max_throttle_change_accel * dt
            )
            final_throttle = last_throttle + accel_press_change

        elif last_throttle <= 0.0 and throttle <= 0.0:
            brake_release_diff = abs(throttle_diff)
            brake_release_change = clip(
                brake_release_diff, 0.0, max_throttle_change_brake_release * dt
            )
            final_throttle = last_throttle + brake_release_change

    elif throttle_diff < 0.0:
        if throttle <= 0.0 and last_throttle >= 0.0:
            accel_release_diff = abs(last_throttle)
            accel_release_change = clip(
                accel_release_diff, 0.0, max_throttle_change_accel_release * dt
            )

            if accel_release_diff >= max_throttle_change_accel_release:
                brake_press_diff = 0.0
                brake_press_change = 0.0
            else:
                brake_press_diff = (
                    max_throttle_change_accel_release - accel_release_change
                )
                brake_press_change = clip(
                    brake_press_diff, 0.0, max_throttle_change_brake * dt
                )

            final_throttle = last_throttle - accel_release_change - brake_press_change

        elif throttle >= 0.0 and last_throttle >= 0.0:
            accel_release_diff = abs(throttle_diff)
            accel_release_change = clip(
                accel_release_diff, 0.0, max_throttle_change_accel_release * dt
            )
            final_throttle = last_throttle - accel_release_change

        elif throttle <= 0.0 and last_throttle <= 0.0:
            brake_press_diff = abs(throttle_diff)
            brake_press_change = clip(
                brake_press_diff, 0.0, max_throttle_change_brake * dt
            )
            final_throttle = last_throttle - brake_press_change

    return final_throttle


def pid_speed_control(target, current, I_term_prev, dt, Kp_v, tauI_v):
    """
    Proportional control for the speed.
    :param target: (float)
    :param current: (float)
    :param I_term_prev: (float)
    :param dt: (float)
    :param Kp_v: (float)
    :param tauI_v: (float)
    :return: (float)
    """

    # error = clip(target - current,-50.0,2.5)
    error = target - current

    P_term = Kp_v * error
    I_term = I_term_prev + (Kp_v / tauI_v) * error * dt

    control_term = P_term + I_term

    return control_term, I_term


def PID_main(target, current, I_term, dir, dt, Kp_v, tauI_v):
    """
    Uses the PID control function & implements windup protection to avoid controller saturation
    :param target: (float)
    :param current: (float)
    :param I_term: (float)
    :param dir: (float)
    :param dt: (float)
    :return: (float, float)
    """

    if dir == "F":
        ai, I_term = pid_speed_control(target, current, I_term, dt, Kp_v, tauI_v)
        # Throttle limits
        throttle_gas_level = clip(ai, -1.0, 1.0)

    elif dir == "R":
        if current >= 0.0:
            ai, I_term = pid_speed_control(-0.1, current, I_term, dt, Kp_v, tauI_v)
            # Throttle limits
            throttle_gas_level = clip(ai, -1.0, 1.0)
        elif current < 0.0:
            ai, I_term = pid_speed_control(target, current, I_term, dt, Kp_v, tauI_v)
            # Throttle limits
            throttle_gas_level = clip(ai, -1.0, 1.0)

    I_term = anti_windup_protection(target, current, throttle_gas_level, ai, I_term)

    return throttle_gas_level, I_term