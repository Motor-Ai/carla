import csv
import sqlite3
import math
from rosidl_runtime_py.utilities import get_message
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 19})
plt.rcParams["figure.figsize"] = (9,8)
from rclpy.serialization import deserialize_message
#import pandas as pd
import numpy as np
#import math
#from bitarray import bitarray
#from bitarray.util import ba2int
import struct

from scipy.signal import savgol_filter

from scipy.signal import butter, lfilter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    '''
    Butterworth digital and analog filter design.
    Returns the filter coefficients for the given order, cutoff & sampling frequency
    Parameters:
    order[type: int]: The order of the filter
    cutoff[type: float]: The critical frequency or frequencies of the filter. 
                         The point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”)
    fs[type: float]: The sampling frequency of the digital system.
    '''
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(x, cutoff, fs, order=5):
    # Calculate filter coefficients for the given order, cutoff & sampling frequency
    b, a = butter_lowpass(cutoff, fs, order=order)

    # Filter a data sequence, x, using a digital filter[IIR or FIR filter] using the inputs
    # b - The numerator coefficient vector in a 1-D sequence.
    # a - The denominator coefficient vector in a 1-D sequence.
    # y = lfilter(b, a, x)
    y = filtfilt(b, a, x)
    return y



class BagFileParser():
    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()

        ## create a message type map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_type = {name_of:type_of for id_of,name_of,type_of in topics_data}
        self.topic_id = {name_of:id_of for id_of,name_of,type_of in topics_data}
        self.topic_msg_message = {name_of:get_message(type_of) for id_of,name_of,type_of in topics_data}

    def __del__(self):
        self.conn.close()

    # Return [(timestamp0, message0), (timestamp1, message1), ...]
    def get_messages(self, topic_name):

        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        # Deserialise all and timestamp them
        return [(timestamp,deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp,data in rows]


def complimentary_filter(a_imu, delta_t, a_ego):
    # Define initial conditions and filter parameters
    a_filtered = 0.0  # Initial filtered acceleration
    tau = 0.005  # Example time constant (adjust as needed)

    # Compute alpha
    alpha = tau / (tau + delta_t)  # delta_t is the time between IMU readings
    #print('Alpha', alpha, '  Delta_T', delta_t)
    
    # Apply complementary filter
    a_filtered = alpha * a_imu + (1 - alpha) * a_ego

    return a_filtered


def moving_average(data, window_size):
    """
    Smooths a 1D array using a moving average filter.
    
    Parameters:
        data (array-like): The input 1D array to be smoothed.
        window_size (int): The size of the moving window.
        
    Returns:
        smoothed (numpy.ndarray): The smoothed array.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid').tolist()




def smooth_array(data, window_length, polynomial_order):
    """
    Smooths a 1D array using the Savitzky-Golay filter.
    
    Parameters:
        data (array-like): The input 1D array to be smoothed.
        window_length (int): The length of the window. Must be an odd integer.
        polynomial_order (int): The order of the polynomial to fit.
        
    Returns:
        smoothed (numpy.ndarray): The smoothed array.
    """
    if window_length % 2 == 0:
        raise ValueError("Window length must be odd.")
    
    return savgol_filter(data, window_length, polynomial_order)


def main():
    bag_file = '/home/arvind/Desktop/mayjunejulydemo/MPC/Motor/Ros_Historic_Core/bags/180424_foxy_7_coupled/180424_foxy_7_coupled_0.db3'

    parser = BagFileParser(bag_file)

    canbus_data = parser.get_messages('/shadow/canbus')

    time_arr = []
    steering_value_arr = []
    brake_value_arr = []
    throttle_value_arr = []
    checksum_arr = []
    speed_arr = []
    accel_arr = []

    for i in range(len(canbus_data)):
        canbus_values = canbus_data[i]
        checksum =  canbus_values[1].checksum
        if checksum == 1:
            timestamp_canbus = canbus_values[1].header.stamp.sec + (canbus_values[1].header.stamp.nanosec)/1000000000.0
            steering_canbus = canbus_values[1].steering
            speed_value = canbus_values[1].speed/3.6
            brake_canbus = canbus_values[1].brake
            throttle_canbus = canbus_values[1].throttle
            accel_canbus = canbus_values[1].accel

            time_arr.append(timestamp_canbus)
            steering_value_arr.append(steering_canbus)
            brake_value_arr.append(brake_canbus)
            throttle_value_arr.append(throttle_canbus)
            speed_arr.append(speed_value)
            accel_arr.append(accel_canbus)
            checksum_arr.append(checksum)

    brake_value_arr = np.array(brake_value_arr)
    throttle_value_arr = np.array(throttle_value_arr)

    #smooth_speed_arr = moving_average(speed_arr, 3)
    #smooth_speed_arr = smooth_array(speed_arr, 21, 2)
    smooth_speed_arr = butter_lowpass_filter(speed_arr, 0.3, 21, order=5)
    smooth_accel_arr = butter_lowpass_filter(accel_arr, 0.3, 7, order=5)

    accel_filt_arr = [smooth_accel_arr[0]]
    accel_v_ego_arr = [smooth_accel_arr[0]]
    for i in range(len(time_arr)):
        if i > 0:
            t_Delta = time_arr[i] - time_arr[i-1]
            v_Delta = smooth_speed_arr[i] - smooth_speed_arr[i-1]
            accel_v_ego = v_Delta/t_Delta
            accel_v_ego_arr.append(accel_v_ego)
            accel_filt_arr.append(complimentary_filter(smooth_accel_arr[i], t_Delta, accel_v_ego))



    plt.figure(1)
    plt.title('IMU Accel vs Time')
    #plt.plot(np.array(time_arr) - time_arr[0], accel_arr, label='IMU_Accel (m/s²)')
    plt.plot(np.array(time_arr) - time_arr[0], accel_v_ego_arr, label='Speed_differential_Accel (m/s²)')
    plt.plot(np.array(time_arr) - time_arr[0], accel_filt_arr, label='Filtered_Accel (m/s²)')
    plt.plot(np.array(time_arr) - time_arr[0], smooth_accel_arr, label='Filtered_IMU_Accel (m/s²)')
    plt.grid()
    plt.legend()
    plt.xlabel('Time(secs)')
    plt.ylabel('Accel (m/s²)')
    #plt.savefig('Steering vs Time figure')
    plt.show()

    plt.figure(2)
    plt.title('Ego Speed vs Time')
    plt.plot(np.array(time_arr) - time_arr[0], speed_arr, label='Ego_Speed (m/s)')
    plt.plot(np.array(time_arr) - time_arr[0], smooth_speed_arr, label='Ego_Speed Smoothed (m/s)')
    plt.grid()
    plt.legend()
    plt.xlabel('Time(secs)')
    plt.ylabel('Ego_Speed (m/s)')
    plt.show()


    plt.figure(3)
    plt.title('Accel vs Velocity - Throttle(blue) & Braking(black)')
    plt.scatter(smooth_speed_arr[throttle_value_arr > 5.0]*3.6, smooth_accel_arr[throttle_value_arr > 5.0], c='b')
    plt.scatter(smooth_speed_arr[brake_value_arr > 5.0]*3.6, smooth_accel_arr[brake_value_arr > 5.0], c='k')
    # for i in range(len(smooth_speed_arr)):
        
    #     if throttle_value_arr[i] > 10.0:
    #         plt.scatter(smooth_speed_arr[i]*3.6,smooth_accel_arr[i], c='b')
    #     if brake_value_arr[i] > 5.0:
    #         plt.scatter(smooth_speed_arr[i]*3.6,smooth_accel_arr[i], c='k')
    plt.grid()
    plt.legend()
    plt.xlabel('Ego Vel(kph)')
    plt.ylabel('Ego Accel (m/s²)')
    plt.show()


if __name__ == '__main__':
    main()
