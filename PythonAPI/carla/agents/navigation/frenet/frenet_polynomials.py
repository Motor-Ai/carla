import numpy as np


class QuarticPolynomial:
    def __init__(self, start_x, start_vel, start_acc, end_vel, end_acc, traj_time):
        # calculate the coefficients of 4th order polynomical equation
        self.a0 = start_x
        self.a1 = start_vel
        self.a2 = start_acc / 2.0
        self.T = traj_time

        # Define Matrices of Equation AX = b
        A = np.array(
            [[3 * (self.T**2), 4 * (self.T**3)], [6 * self.T, 12 * (self.T**2)]]
        )

        b = np.array([end_vel - self.a1 - 2 * self.a2 * self.T, end_acc - 2 * self.a2])

        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]

    def calc_path_derivatives(self, dt):
        ## Calc time arr
        t_arr = np.linspace(0.0, self.T - dt, int(self.T / dt))

        ## Eq(1) --> a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4
        x = (
            self.a0
            + self.a1 * t_arr
            + self.a2 * (t_arr**2)
            + self.a3 * (t_arr**3)
            + self.a4 * (t_arr**4)
        )
        x = x.tolist()

        ## single time-differential of Eq(1)
        dx = (
            self.a1
            + 2 * self.a2 * t_arr
            + 3 * self.a3 * (t_arr**2)
            + 4 * self.a4 * (t_arr**3)
        )
        dx = dx.tolist()

        ## second time-differential of Eq(1QuarticPolynomial)
        ddx = 2 * self.a2 + 6 * self.a3 * t_arr + 12 * self.a4 * (t_arr**2)
        ddx = ddx.tolist()

        ## third time-differential of Eq(1)
        dddx = 6 * self.a3 + 24 * self.a4 * t_arr
        dddx = dddx.tolist()

        return x, dx, ddx, dddx


class QuinticPolynomial:
    def __init__(
        self, start_x, start_vel, start_acc, end_x, end_vel, end_acc, traj_time
    ):
        # calculate the coefficients of 5th order polynomical equation
        self.a0 = start_x
        self.a1 = start_vel
        self.a2 = start_acc / 2.0
        self.T = traj_time

        # Define Matrices of Equation AX = b
        A = np.array(
            [
                [self.T**3, self.T**4, self.T**5],
                [3 * (self.T**2), 4 * (self.T**3), 5 * (self.T**4)],
                [6 * self.T, 12 * (self.T**2), 20 * (self.T**3)],
            ]
        )

        b = np.array(
            [
                end_x - self.a0 - self.a1 * self.T - self.a2 * (self.T**2),
                end_vel - self.a1 - 2 * self.a2 * self.T,
                end_acc - 2 * self.a2,
            ]
        )

        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_path_derivatives(self, dt):
        ## Calc time arr
        t_arr = np.linspace(0.0, self.T - dt, int(self.T / dt))

        ## Eq(1) --> a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        x = (
            self.a0
            + self.a1 * t_arr
            + self.a2 * (t_arr**2)
            + self.a3 * (t_arr**3)
            + self.a4 * (t_arr**4)
            + self.a5 * (t_arr**5)
        )
        x = x.tolist()

        ## single time-differential of Eq(1)
        dx = (
            self.a1
            + 2 * self.a2 * t_arr
            + 3 * self.a3 * (t_arr**2)
            + 4 * self.a4 * (t_arr**3)
            + 5 * self.a5 * (t_arr**4)
        )
        dx = dx.tolist()

        ## second time-differential of Eq(1)
        ddx = (
            2 * self.a2
            + 6 * self.a3 * t_arr
            + 12 * self.a4 * (t_arr**2)
            + 20 * self.a5 * (t_arr**3)
        )
        ddx = ddx.tolist()

        ## third time-differential of Eq(1)
        dddx = 6 * self.a3 + 24 * self.a4 * t_arr + 60 * self.a5 * (t_arr**2)
        dddx = dddx.tolist()

        return x, dx, ddx, dddx


class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.c = []

        self.dynamic_intersection_area = []
        self.dynamic_collision = []