import json
import math
import os
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('fivethirtyeight')

from scipy.signal import medfilt, butter, filtfilt

# import fft(Fast Fourier Transform) function to convert a signal from time domain to
# frequency domain (output :is a numpy array contains signal's amplitudes of each frequency component)
from scipy.fftpack import fft

# import fftfreq function to generate frequencies related to frequency components mentioned above
from scipy.fftpack import fftfreq

# import ifft function (inverse fft) inverse the conversion
from scipy.fftpack import ifft

# import fftpack to use all fft functions
from scipy import fftpack
from numpy.fft import *

from sklearn.preprocessing import StandardScaler
import joblib

# ================================================ GET META VARIABLES ================================================

from . import load
from . import utils

CUR_DIR = utils.get_full_path_directory(mode="main")
DATA_DIR = utils.get_full_path_directory(mode="data")
CONFIG_DIR = utils.get_full_path_directory(mode="configs")
LOGS_DIR = utils.get_full_path_directory(mode="logs")
TEST_DIR = utils.get_full_path_directory(mode="tests")

with open(f"{CONFIG_DIR}/default.json", "r") as f:
    params = json.load(f)["meta_params"]
    sampling_freq = params["sampling_freq"]
    window_size = params["window_size"]
    prefix = params["prefix"]
    n_splits = params["n_splits"]

logger = getLogger(f"{LOGS_DIR}/{prefix}_freq_{sampling_freq}_ws_{window_size}_split_{n_splits}.log")

freq1 = 0.3  
# freq1=0.3 hertz [Hz] the cutoff frequency between the DC components [0,0.3] and the body components[0.3,20]hz
freq2 = 20  # freq2=20 Hz the cutoff frequency between the body components [0.3,20] hz and the high frequency noise
# components [20,25]hz

# d(signal)/dt : the Derivative
dt = 1/sampling_freq

# ==================================================== Functions for filtering the signals
# ====================================================

def apply_filter(signal, filter="median", window=3):
    """A denoising filter is applied to remove noise in signals.
    Args:
        signal (numpy array 1D (one column)): Raw signal
        filter (str, default='median'): Filter name is chosen from 'mean', 'median', or 'butterworth'
        window (int, default=5): Length of filter
    Returns:
        signal (pd.DataFrame): Filtered signal
    See Also:
        'butterworth' applies a 3rd order low-pass Butterworth filter with a corner frequency of 20 Hz.
    """
    if filter == "mean":
        signal = signal.rolling(window=window, center=True, min_periods=1).mean()
    elif filter == "median":
        signal = signal.rolling(window=window, center=True, min_periods=1).median()
    elif filter == "butterworth":
        fc = 20  # cutoff frequency
        w = float(fc)/(50.0/2.0)  # Normalize the frequency
        b, a = butter(3, w, "low")  # 3rd order low-pass Butterworth filter
        signal = filtfilt(b, a, signal, axis=0)
    return signal


def components_selection_one_signal(t_signal, freq1, freq2):
    """
    Args: 
        t_signal:1D numpy array (time domain signal);
    Returns:
        tuple(total_component,t_DC_component , t_body_component, t_noise) type(1D array,1D array, 1D array)
    Cases to discuss:
        if the t_signal is an acceleration signal then the t_DC_component is the gravity component [Grav_acc]
        if the t_signal is a gyro signal then the t_DC_component is not useful
        t_noise component is not useful
        if the t_signal is an acceleration signal then the t_body_component is the body's acceleration component
        [Body_acc]
        if the t_signal is a gyro signal then the t_body_component is the body's angular velocity component [
        Body_gyro]
    """
    t_signal = np.array(t_signal)
    t_signal_length = len(t_signal)  # number of points in a t_signal
    
    # the t_signal in frequency domain after applying fft
    f_signal = fft(t_signal)  # 1D numpy array contains complex values (in C)
    
    # generate frequencies associated to f_signal complex values
    freqs = np.array(fftfreq(t_signal_length, d=1/float(sampling_freq)))  # frequency values between [-25hz:+25hz]
    
    # matplotlib histogram
#     plt.figure(figsize=(20,4))
#     plt.hist(freqs, color = 'blue', edgecolor = 'black', bins = 200)
#     seaborn histogram
#     sns.distplot(freqs, hist=True, kde=False, 
#     bins=200, color = 'blue',
#     hist_kws={'edgecolor':'black'})
#     # Add labels
#     plt.title('Histogram of Frequencies')
#     plt.xlabel('Frequency')
#     plt.ylabel('Records')
    
    # DC_component: f_signal values having freq between [-0.3 hz to 0 hz] and from [0 hz to 0.3hz] (-0.3 and 0.3 are
    # included)
    # noise components: f_signal values having freq between [-25 hz to 20 hz[ and from ] 20 hz to 25 hz] 
    #                   (-25 and 25 hz included 20hz and -20hz not included)
    # selecting body_component: f_signal values having freq between [-20 hz to -0.3 hz] and from [0.3 hz to 20 hz] 
    #                           (-0.3 and 0.3 not included , -20hz and 20 hz included)
    f_DC_signal = []  # DC_component in freq domain
    f_body_signal = []  # body component in freq domain numpy.append(a, a[0])
    f_noise_signal = []  # noise in freq domain
    for i in range(len(freqs)):
        freq = freqs[i]
        value = f_signal[i]
        if abs(freq) > freq1:  # testing if freq is outside DC_component frequency ranges
            f_DC_signal.append(float(
                    0))  # add 0 to  the  list if it was the case (the value should not be added)
        else:  # if freq is inside DC_component frequency ranges
            f_DC_signal.append(value)  # add f_signal value to f_DC_signal list
        
        if abs(freq) <= freq2:  # testing if freq is outside noise frequency ranges
            f_noise_signal.append(float(0))  # # add 0 to  f_noise_signal list if it was the case
        
        else:  # if freq is inside noise frequency ranges
            f_noise_signal.append(value)  # add f_signal value to f_noise_signal
        
        # Selecting body_component values 
        if abs(freq) <= freq1 or abs(freq) > freq2:  # testing if freq is outside Body_component frequency ranges
            f_body_signal.append(float(0))  # add 0 to  f_body_signal list
        
        else:  # if freq is inside Body_component frequency ranges
            f_body_signal.append(value)  # add f_signal value to f_body_signal list
    
    # -------------------------- Inverse the transformation of signals in freq domain --------------------------
    # applying the inverse fft(ifft) to signals in freq domain and put them in float format
    t_DC_component = ifft(np.array(f_DC_signal)).real
    t_body_component = ifft(np.array(f_body_signal)).real
    t_noise = ifft(np.array(f_noise_signal)).real
    total_component = t_signal-t_noise  # extracting the total component(filtered from noise)
    #  by subtracting noise from t_signal (the original signal).
    # return outputs mentioned earlier
    return (total_component, t_DC_component, t_body_component, t_noise)


def mag_3_signals(x, y, z):  # Euclidian magnitude
    return [math.sqrt((x[i] ** 2+y[i] ** 2+z[i] ** 2)) for i in range(len(x))]


def verify_gravity(user_id, exp_id, raw_dic, LOG_DIR, pdf_write=None):
    key = f"{user_id}_{exp_id}"
    if key not in raw_dic.keys():
        return
    acc_x = np.array(raw_dic[key]['AccX'])
    acc_y = np.array(raw_dic[key]['AccY'])
    acc_z = np.array(raw_dic[key]['AccZ'])
    
    # apply the filtering method to acc_[X,Y,Z] and store gravity components
    grav_acc_X = components_selection_one_signal(acc_x, freq1, freq2)[1]
    grav_acc_Y = components_selection_one_signal(acc_y, freq1, freq2)[1]
    grav_acc_Z = components_selection_one_signal(acc_z, freq1, freq2)[1]
    
    # calculating gravity magnitude signal
    grav_acc_mag = mag_3_signals(grav_acc_X, grav_acc_Y, grav_acc_Z)
    mean_value = str(np.array(grav_acc_mag).mean())[0:5]
    x_labels = f'Time in seconds[s]'
    y_labels = f'Gravity amplitude'
    title = f'Euclidian magnitude of gravity 3-axial signals [Mean value: {mean_value}]'
    legend = key+' grav_acc_mag'  # set the figure's legend
    
    visualize_signal(grav_acc_mag, x_labels, y_labels, title, legend, LOG_DIR, name=f"Gravity_{user_id}_{exp_id}",
                     pdf_write=pdf_write)


# ====================================================== VISUALIZATION functions for signal processing =================
chosen_cols = ["AccX", "AccY", "AccZ", "GyroYaw", "GyroPitch", "GyroRoll"]

def visualize_signal(signal, x_labels, y_labels, title, legend, LOG_DIR, name="SignalImage", pdf_write=None):
    """Visualization for one single signal
    Args: 
        signal: 1D column
        x_labels: the X axis info (figure)
        y_labels: the Y axis info (figure)
        title: figure's title
        legend : figure's legend
        LOG_DIR: logging directory
        name: image name
        pdf_write: PdfPages object
    Returns: 
        None
    """
    # Define the figure's dimensions
    fig = plt.figure(figsize=(20, 4))
    
    # convert row numbers in time durations
    time = [1/float(sampling_freq)*i for i in range(len(signal))]
    
    # plotting the signal
    plt.plot(time, signal, label=legend, linewidth=2)
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.title(title)
    plt.legend(loc="upper left")
    if pdf_write:
        pdf_write.savefig(fig, bbox_inches='tight', pad_inches=0.3)
    else:
        fig.savefig(os.path.join(LOG_DIR, f"{name}.pdf"), bbox_inches='tight', pad_inches=0.3)


def visualize_triaxial_filtered_signals(user_id, exp_id, raw_dic, LOG_DIR, name="SignalImage", pdf_write=None,
                                        target_columns=chosen_cols):
    """Visualization for 3-axial signals
    Args:
        user_id: user ID
        exp_id: experiment ID
        raw_dic: the Y axis info (figure)
        LOG_DIR: logging directory
        name: image name
        pdf_write: PdfPages object
    Returns:
        None

    """
    key = f"{user_id}_{exp_id}"
    if key not in raw_dic.keys():
        return
    data_frame = raw_dic[key]
    
    for col in target_columns:
        time = np.array(data_frame["time"])
        signal = np.array(data_frame[col])
        
        med_filtered_signal = apply_filter(signal, filter="median")
        med_filtered_signal_diff = med_filtered_signal-signal
        butter_filtered_signal = apply_filter(med_filtered_signal, filter="butterworth")
        butter_filtered_signal_diff = butter_filtered_signal-med_filtered_signal
        
        fig = plt.figure(figsize=(24, 12))
        layout = (3, 3)
        original_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        med_ax = plt.subplot2grid(layout, (1, 0))
        med_diff_ax = plt.subplot2grid(layout, (1, 1))
        butter_ax = plt.subplot2grid(layout, (2, 0))
        butter_diff_ax = plt.subplot2grid(layout, (2, 1))
        
        legend1 = key+f' {col.upper()} original '
        legend2 = key+f' {col.upper()} med_filterd'
        legend3 = key+f' {col.upper()} med_filterd_diff'
        legend4 = key+f' {col.upper()} butterworth_filterd'
        legend5 = key+f' {col.upper()} butterworth_filterd_diff'
        
        x_labels = 'Time in seconds[s]'
        y_labels_1 = 'Acceleration in 1g'
        y_labels_2 = 'Angular Velocity[rad/s]'
        
        title1 = f'Original {col.upper()} signals for all activities performed by user {user_id} in experience {exp_id}'
        title2 = f'median_filter(original_signal)'
        title3 = f'(median_filtered)-(original_signal)'
        title4 = f'butterworth_filter(median_filtered)'
        title5 = f'(butterworth_filtered)-(median_filtered)'
        
        for sig, ax, legend, title in zip(
                [signal, med_filtered_signal, med_filtered_signal_diff, butter_filtered_signal,butter_filtered_signal_diff],
                [original_ax, med_ax, med_diff_ax, butter_ax, butter_diff_ax],
                [legend1, legend2, legend3, legend4, legend5],
                [title1, title2, title3, title4, title5]
        ):
            ax.plot(time, sig, label=legend, linewidth=2)
            y_labels = y_labels_1 if "Acc" in col else y_labels_2
            ax.set_xlabel(x_labels)
            ax.set_ylabel(y_labels)
            ax.set_title(title)
            ax.legend(loc="upper left")
            plt.tight_layout()
        if pdf_write:
            pdf_write.savefig(fig, bbox_inches='tight', pad_inches=0.3)
        else:
            fig.savefig(os.path.join(LOG_DIR, f"{name}.pdf"), bbox_inches='tight', pad_inches=0.3)
    plt.close()


# ==================================================== Functions for creating Jerk and time signal functions ===========
def jerk_one_signal(signal):
    """Returns Jerk signal: the rate of changes in accelerator in time-domain: jerk(signal(x0)) is equal to (signal(
    x0+dx)-signal(x0))/dt
       Where: signal(x0+dx)=signal[index[x0]+1] and signal(x0)=signal[index[x0]]
    Args: 
        signal: 1D array signal
    Returns: 
        1D array of Jerk signals
    """
    return np.array([(signal[i+1]-signal[i])/dt for i in range(len(signal)-1)])


def create_time_sig_dic(raw_dic, mode="complex"):
    """Create time-signal dictionary contains multiple dataframes where each dataframe is composed into many
    components [body, gravity, jerk]
    Args: 
        mode(string): if 'simple' indicates that we just want to do filtering but not component selection 
                      if 'complex' indicates that we want to do component selection into body, gravity
    Returns: 
        Composed DataFrames with many elements (body, gravity, jerk)
    """
    raw_dic_keys = sorted(raw_dic.keys())  # sorting dataframes' keys
    time_sig_dic = {}
    target_columns = ['AccX', 'AccY', 'AccZ', 'GyroYaw', 'GyroPitch', 'GyroRoll']
    for key in raw_dic_keys:
        raw_df = raw_dic[key].copy()
        time_sig_df = pd.DataFrame()
        raw_df_target = [i for i in raw_df.columns if i in target_columns]
        for column in raw_df_target:
            t_signal = np.array(raw_df[column].copy())
            #===================================================================
            # REMOVE THIS (CONSIDER THIS)
            if mode == "simple":
                new_columns_ordered = target_columns
                # t_signal = apply_filter(t_signal, filter="butterworth")
                time_sig_df[column] = t_signal
            
            #===================================================================
            elif mode == "complex":
                new_columns_ordered = ['t_body_AccX', 't_body_AccY', 't_body_AccZ',
                                       't_grav_AccX', 't_grav_AccY', 't_grav_AccZ',
                                       't_body_jerk_AccX', 't_body_jerk_AccY', 't_body_jerk_AccZ',
                                       't_body_GyroYaw', 't_body_GyroPitch', 't_body_GyroRoll',
                                       't_body_jerk_GyroYaw', 't_body_jerk_GyroPitch', 't_body_jerk_GyroRoll']
            #===================================================================
                if 'Acc' in column:
                    _, grav_acc, body_acc, _ = components_selection_one_signal(t_signal, freq1, freq2)
                    body_acc_jerk = jerk_one_signal(body_acc)
                    # store signal in time_sig_dataframe and delete the last value of each column 
                    # jerked signal will have the original length-1(due to jerking)
                    time_sig_df['t_body_'+column] = body_acc[:-1]
                    time_sig_df['t_grav_'+column] = grav_acc[:-1]
                    # store t_body_acc_jerk signal with the appropriate axis selected from the column name
                    time_sig_df['t_body_jerk_'+column] = body_acc_jerk
                    
            #===================================================================
                elif 'Gyro' in column:
                    _, _, body_gyro, _ = components_selection_one_signal(t_signal, freq1, freq2)
                    body_gyro_jerk = jerk_one_signal(body_gyro)
                    # store signal in time_sig_dataframe and delete the last value of each column 
                    # jerked signal will have the original length-1(due to jerking)
                    time_sig_df['t_body_'+column] = body_gyro[:-1]
                    time_sig_df['t_body_jerk_'+column] = body_gyro_jerk
        
        time_sig_df = time_sig_df[new_columns_ordered]
        
        if mode == "complex":
            for i in range(0, 15, 3):  # range defined based on new_columns_ordered list columns
                mag_col_name = utils.remove_direction_feature(new_columns_ordered[i])+'_mag'
                col0 = np.array(time_sig_df[new_columns_ordered[i]])
                col1 = time_sig_df[new_columns_ordered[i+1]]
                col2 = time_sig_df[new_columns_ordered[i+2]]
                mag_signal = mag_3_signals(col0, col1, col2)
                time_sig_df[mag_col_name] = mag_signal
        time_sig_dic[key] = time_sig_df
    return time_sig_dic


# ================================================================Functions for Frequency-magnitude ====================
def fast_fourier_transform_one_signal(t_signal):
    """
    Args: 
        time signal 1D array
    Returns: 
        amplitude of fft components 1D array having the same length as the Input
    """
    # apply fast fourier transform to the t_signal
    complex_f_signal = fftpack.fft(t_signal)
    
    # compute the amplitude each complex number
    amplitude_f_signal = np.abs(complex_f_signal)
    
    # return the amplitude
    return amplitude_f_signal


def fast_fourier_transform(t_window):
    """
    Args: 
        A DataFrame body and gravity columns (won't be transformed)
    Returns: 
        A DataFrame with 16 frequency signal (16 columns)
    """
    f_window = pd.DataFrame()  # create an empty dataframe will include frequency domain signals of window
    
    for column in t_window.columns:  # iterating over time domain window columns(signals)
        if 'grav' not in column:  # verify if time domain signal is not related to gravity components
            t_signal = np.array(t_window[column])  # convert the column to a 1D numpy array
            f_signal = np.apply_along_axis(fast_fourier_transform_one_signal, 0,
                                           t_signal)  # apply the function defined above to the column
            f_window[
                "f_"+column[2:]] = f_signal  # storing the frequency signal in f_window with an appropriate column name
    
    return f_window  # return the frequency domain window
