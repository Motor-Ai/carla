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