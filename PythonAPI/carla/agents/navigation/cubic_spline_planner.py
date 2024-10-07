"""
cubic spline planner

Author: Atsushi Sakai

"""
import math
import numpy as np
import bisect


class Spline:
    u"""
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        u"""
        Calc position

        if t is outside of the input x, return None

        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        u"""
        Calc first derivative

        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        u"""
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        u"""
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        u"""
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        u"""
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        #  print(B)
        return B


class Spline2D:
    u"""
    2D Cubic Spline class

    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2)
                   for (idx, idy) in zip(dx, dy)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        u"""
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        u"""
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        return k

    def calc_yaw(self, s):
        u"""
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw


class Spline3D:
    """
    3D Cubic Spline class
    """

    def __init__(self, x, y, z):
        self.s = self.__calc_s(x, y, z)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)
        self.sz = Spline(self.s, z)

    def __calc_s(self, x, y, z):
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2 + idz ** 2) for (idx, idy, idz) in zip(dx, dy, dz)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        u"""
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)
        z = self.sz.calc(s)
        return x, y, z

    def calc_curvature(self, s):
        u"""
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        return k

    def calc_yaw(self, s):
        u"""
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw

    def calc_pitch(self, s):
        """
        calc pitch - this function needs to be double checked
        """
        dx = self.sx.calcd(s)
        dz = self.sz.calcd(s)
        pitch = math.atan2(dz, dx)
        return pitch


def calc_spline_course(x, y, ds=0.1):
    sp = Spline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s

import numpy as np
from scipy.interpolate import splprep, splev
import scipy.interpolate as interpolate



def get_heading_arr(x_arr, y_arr):
    """
    Calculate heading of the input path
    """
    if len(x_arr) > 1:
        yaw_arr = np.arctan2(np.diff(y_arr), np.diff(x_arr))
        yaw_arr = np.append(yaw_arr, yaw_arr[-1])
    else:
        yaw_arr = np.array([])
    return yaw_arr


def get_curvature_arr(x_arr, y_arr):
    """
    Calculate curvature of the input path
    """
    if len(x_arr) > 5:
        # Single differential
        dx_dt = np.gradient(x_arr)
        dy_dt = np.gradient(y_arr)
        ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

        # Double differential
        d2s_dt2 = np.gradient(ds_dt)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = -1.0*(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    else:
        curvature = np.array([])

    return curvature


def eliminate_duplicates(arr):
    repeat_vals = (arr[:-1]==arr[1:]).sum(axis = 1)
    idx = np.where(repeat_vals!=2)[0] + 1
    return arr[idx]


def filter_input(x_arr, y_arr, dl):
    """
    Filters out repetitive & consecutive points with distances lesser that define 'dl' resolution
    """
    xy_inp = np.transpose(np.array([x_arr, y_arr]))
    ## remove unique points
    xy_inp_unique = eliminate_duplicates(xy_inp)

    try:
        x_arr_unique = xy_inp_unique[:, 0]
        y_arr_unique = xy_inp_unique[:, 1]
    except:
        x_arr_unique = np.array([])
        y_arr_unique = np.array([])

    if len(x_arr_unique) > 1:
        ##Find consecutive distance
        x_arr_diff = np.diff(x_arr_unique)
        y_arr_diff = np.diff(y_arr_unique)
        dist_btw = np.round(np.hypot(x_arr_diff, y_arr_diff), 2)
        idx_remove = np.where(dist_btw <= dl)[0] + 1

        ##Delete consecutive points with distances lesser that defined 'dl' resolution
        x_arr_final = np.delete(x_arr_unique, idx_remove)
        y_arr_final = np.delete(y_arr_unique, idx_remove)

        return x_arr_final, y_arr_final

    else:
        return x_arr_unique, y_arr_unique


def calc_spline_course(x_arr, y_arr, input_res, output_res):
    try:
        ##Filtering step to remove repetitive & too close points
        x, y = filter_input(x_arr, y_arr, output_res)

        #create spline function
        f, u = splprep([x_arr, y_arr],s=0)

        #calculate resolution ratio
        res_ratio = int(input_res/output_res)

        #create interpolated lists of points
        x_spline, y_spline = splev(np.linspace(0, 1, int(len(x_arr)*res_ratio)), f)

        yaw_spline = get_heading_arr(x_spline, y_spline)

        curvature_spline = get_curvature_arr(x_spline, y_spline)

        return x_spline, y_spline, yaw_spline, curvature_spline
    except Exception as e:
        print("calc_spline_course (spline.py) : Extrapolation step invalid input")
        print(e)
        return [], [], [], []



def calc_spline_course_2(x_arr, y_arr, path_len, output_res):
    try:
        ##Filtering step to remove repetitive & too close points
        x, y = filter_input(x_arr, y_arr, output_res)

        #create spline function
        f, u = splprep([x_arr, y_arr])

        #create interpolated lists of points
        x_spline, y_spline = splev(np.linspace(0, 1, int(path_len/output_res)), f)

        yaw_spline = get_heading_arr(x_spline, y_spline)

        curvature_spline = get_curvature_arr(x_spline, y_spline)

        return x_spline.tolist(), y_spline.tolist(), yaw_spline.tolist(), curvature_spline.tolist()
    except Exception as e:
        print("calc_spline_course (spline.py) : Extrapolation step invalid input")
        print(e)
        return [], [], [], []


def approximate_b_spline_path(x: list,
                              y: list,
                              n_path_points: int,
                              degree: int = 3,
                              s=None,
                              ) -> tuple:
    """
    Approximate points with a B-Spline path

    Parameters
    ----------
    x : array_like
        x position list of approximated points
    y : array_like
        y position list of approximated points
    n_path_points : int
        number of path points
    degree : int, optional
        B Spline curve degree. Must be 2<= k <= 5. Default: 3.
    s : int, optional
        smoothing parameter. If this value is bigger, the path will be
        smoother, but it will be less accurate. If this value is smaller,
        the path will be more accurate, but it will be less smooth.
        When `s` is 0, it is equivalent to the interpolation. Default is None,
        in this case `s` will be `len(x)`.

    Returns
    -------
    x : array
        x positions of the result path
    y : array
        y positions of the result path
    heading : array
        heading of the result path
    curvature : array
        curvature of the result path

    """
    distances = _calc_distance_vector(x, y)

    spl_i_x = interpolate.UnivariateSpline(distances, x, k=degree, s=s)
    spl_i_y = interpolate.UnivariateSpline(distances, y, k=degree, s=s)

    sampled = np.linspace(0.0, distances[-1], n_path_points)
    return _evaluate_spline(sampled, spl_i_x, spl_i_y)


def _calc_distance_vector(x, y):
    dx, dy = np.diff(x), np.diff(y)
    distances = np.cumsum([np.hypot(idx, idy) for idx, idy in zip(dx, dy)])
    distances = np.concatenate(([0.0], distances))
    distances /= distances[-1]
    return distances


def _evaluate_spline(sampled, spl_i_x, spl_i_y):
    x = spl_i_x(sampled)
    y = spl_i_y(sampled)
    return np.array(x), np.array(y)


def calc_bspline_course(x_arr, y_arr, input_res, output_res):
    try:
        ##Filtering step to remove repetitive & too close points
        x, y = filter_input(x_arr, y_arr, output_res)

        #calculate resolution ratio
        res_ratio = int(input_res/output_res)

        #create interpolated lists of points
        x_spline, y_spline = approximate_b_spline_path(x_arr, y_arr, int(len(x_arr)*res_ratio), s=0.5)

        yaw_spline = get_heading_arr(x_spline, y_spline)

        curvature_spline = get_curvature_arr(x_spline, y_spline)

        return x_spline, y_spline, yaw_spline, curvature_spline
    except Exception as e:
        print("calc_spline_course (spline.py) : Extrapolation step invalid input")
        print(e)
        return [], [], [], []


def calc_bspline_course_2(x_arr, y_arr, path_len, output_res):
    try:
        ##Filtering step to remove repetitive & too close points
        x, y = filter_input(x_arr, y_arr, output_res)
        
        #create interpolated lists of points
        x_spline, y_spline = approximate_b_spline_path(x_arr, y_arr, int(path_len/output_res), s=0.5)

        yaw_spline = get_heading_arr(x_spline, y_spline)

        curvature_spline = get_curvature_arr(x_spline, y_spline)

        return x_spline.tolist(), y_spline.tolist(), yaw_spline.tolist(), curvature_spline.tolist()
    except Exception as e:
        print("calc_spline_course (spline.py) : Extrapolation step invalid input")
        print(e)
        return [], [], [], []
    

def main():
    print("Spline 2D test")
    import matplotlib.pyplot as plt
    x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]

    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    flg, ax = plt.subplots(1)
    plt.plot(x, y, "xb", label="input")
    plt.plot(rx, ry, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    flg, ax = plt.subplots(1)
    plt.plot(s, [math.degrees(iyaw) for iyaw in ryaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    flg, ax = plt.subplots(1)
    plt.plot(s, rk, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.show()


if __name__ == '__main__':
    main()
