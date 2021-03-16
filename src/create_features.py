import json
import math
from logging import getLogger
from math import acos, sqrt  # inverse of cosin function; square root function

import numpy as np
import pandas as pd
from scipy.fftpack import fftfreq

# import the entropy function; import interquartile range function (Q3(column)-Q1(column))
from scipy.stats import entropy, iqr as IQR, kurtosis, pearsonr, skew
from statsmodels.robust import mad as median_deviation  # import the median deviation function

from . import utils

# ================================================ GET META VARIABLES ==================================================

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


def remove_direction_feature(name):
    sub_names = name.split("_")
    if "Acc" in name:
        return name.replace(sub_names[-1], "Acc")
    elif "Gyro" in name:
        return name.replace(sub_names[-1], "Gyro")
    else:
        return name


def assert_middle_feature(name, feature):
    sub_names = name.split("_")
    return "_".join(sub_names[:-1])+feature+"_"+sub_names[-1]


# f_W00000_Jeff_wk4_act02
def extract_info_window_key(key, mode="user"):
    names = key.split("_")
    if mode == "window":
        return names[1]
    elif mode == "user":
        return names[2]
    elif mode == "experiment":
        return names[3]
    elif mode == "activity":
        return names[-1][-2:]


# ================================================ Functions for COMMON features =======================================
# df is dataframe contains 3 columns (3 axial signals X,Y,Z)
# mean
def mean_axial(df):
    array = np.array(df)  # convert dataframe into 2D numpy array for efficiency
    mean_vector = list(array.mean(axis=0))  # calculate the mean value of each column
    return mean_vector  # return mean vector


# std
def std_axial(df):
    array = np.array(df)  # convert dataframe into 2D numpy array for efficiency
    std_vector = list(array.std(axis=0))  # calculate the standard deviation value of each column
    return std_vector


# mad
def mad_axial(df):
    array = np.array(df)  # convert dataframe into 2D numpy array for efficiency
    mad_vector = list(median_deviation(array, axis=0))  # calculate the median deviation value of each column
    return mad_vector


# max
def max_axial(df):
    array = np.array(df)  # convert dataframe into 2D numpy array for efficiency
    max_vector = list(array.max(axis=0))  # calculate the max value of each column
    return max_vector


# min
def min_axial(df):
    array = np.array(df)  # convert dataframe into 2D numpy array for efficiency
    min_vector = list(array.min(axis=0))  # calculate the min value of each column
    return min_vector


# IQR
def IQR_axial(df):
    array = np.array(df)  # convert dataframe into 2D numpy array for efficiency
    IQR_vector = list(np.apply_along_axis(IQR, 0, array))  # calculate the inter quartile range value of each column
    return IQR_vector


# Entropy
def entropy_axial(df):
    array = np.array(df)  # convert dataframe into 2D numpy array for efficiency
    entropy_vector = list(np.apply_along_axis(entropy, 0, abs(array)))  # calculate the entropy value of each column
    return entropy_vector


# --------------------------------------------MAGNITUDE functions--------------------------------------------
# mag column : is one column contains one mag signal values
# same features mentioned above were calculated for each column

# mean
def mean_mag(mag_column):
    array = np.array(mag_column)
    mean_value = float(array.mean())
    return mean_value


# std: standard deviation of mag column
def std_mag(mag_column):
    array = np.array(mag_column)
    std_value = float(array.std())  # std value
    return std_value


# mad: median deviation
def mad_mag(mag_column):
    array = np.array(mag_column)
    mad_value = float(median_deviation(array))  # median deviation value of mag_column
    return mad_value


# max
def max_mag(mag_column):
    array = np.array(mag_column)
    max_value = float(array.max())  # max value
    return max_value


# min
def min_mag(mag_column):
    array = np.array(mag_column)
    min_value = float(array.min())  # min value
    return min_value


# IQR
def IQR_mag(mag_column):
    array = np.array(mag_column)
    IQR_value = float(IQR(array))  # Q3(column)-Q1(column)
    return IQR_value


# Entropy
def entropy_mag(mag_column):
    array = np.array(mag_column)
    entropy_value = float(entropy(array))  # entropy signal
    return entropy_value


# ================================================ Functions for TIME features==========================================
# --------------------------------------------List of 3-AXIAL FEATURES--------------------------------------------
# df is dataframe contains 3 columns (3 axial signals X,Y,Z)

# sma
def t_sma_axial(df):
    array = np.array(df)
    sma_axial = float(abs(array).sum())/float(3)  # sum of areas under each signal
    return sma_axial  # return sma value


# energy
def t_energy_axial(df):
    array = np.array(df)
    energy_vector = list((array ** 2).sum(axis=0))  # energy value of each df column
    return energy_vector  # return energy vector energy_X,energy_Y,energy_Z


# correlation
def t_corr_axial(df):  # it returns 3 correlation features per each 3-axial signals in  time_window
    array = np.array(df)
    Corr_X_Y = float(pearsonr(array[:, 0], array[:, 1])[0])  # correlation value between signal_X and signal_Y
    Corr_X_Z = float(pearsonr(array[:, 0], array[:, 2])[0])  # correlation value between signal_X and signal_Z
    Corr_Y_Z = float(pearsonr(array[:, 1], array[:, 2])[0])  # correlation value between signal_Y and signal_Z
    corr_vector = [Corr_X_Y, Corr_X_Z, Corr_Y_Z]  # put correlation values in list
    return corr_vector


# --------------------------------------------List of 3-AXIAL FEATURE
# generation--------------------------------------------
def t_axial_features_generation(t_window):
    # select axial columns : the first 15 columns
    axial_columns = t_window.columns[0:15]
    # select axial columns in a dataframe
    axial_df = t_window[axial_columns]
    # a list will contain all axial features values resulted from applying:
    # common axial features functions and time axial features functions to all time domain signals in t_window
    t_axial_features = []
    for col in range(0, 15, 3):
        df = axial_df[axial_columns[col:col+3]]  # select each group of 3-axial signal: signal_name[X,Y,Z]
        # apply all common axial features functions and time axial features functions to each 3-axial signals dataframe
        mean_vector = mean_axial(df)  # 3values
        std_vector = std_axial(df)  # 3 values
        mad_vector = mad_axial(df)  # 3 values
        max_vector = max_axial(df)  # 3 values
        min_vector = min_axial(df)  # 3 values
        sma_value = t_sma_axial(df)  # 1 value
        energy_vector = t_energy_axial(df)  # 3 values
        IQR_vector = IQR_axial(df)  # 3 values
        entropy_vector = entropy_axial(df)  # 3 values
        
        # corr_vector = t_corr_axial(df)  # 3 values
        
        t_3axial_vector = mean_vector+std_vector+mad_vector+max_vector+min_vector+[sma_value]+energy_vector+IQR_vector+entropy_vector
        
        # +corr_vector
        
        # append these features to the global list of features
        t_axial_features = t_axial_features+t_3axial_vector
    return t_axial_features


# -------------------------------------------- Functions for 3-AXIAL-MAGNITUDE-TIME features----------------------------
# Functions used to generate time magnitude features

# sma: signal magnitude area
def t_sma_mag(mag_column):
    array = np.array(mag_column)
    sma_mag = float(abs(array).sum())  # signal magnitude area of one mag column
    return sma_mag


# energy
def t_energy_mag(mag_column):
    array = np.array(mag_column)
    energy_value = float((array ** 2).sum())  # energy of the mag signal
    return energy_value


# --------------------------------------------List of 3-AXIAL FEATURE generation----------------------------------------
def t_mag_features_generation(t_window):
    mag_columns = t_window.columns[15:]  # mag columns' names
    mag_columns = t_window[mag_columns]  # mag data frame
    t_mag_features = []  # a global list will contain all time domain magnitude features
    for col in mag_columns:  # iterate throw each mag column
        mean_value = mean_mag(mag_columns[col])  # 1 value
        std_value = std_mag(mag_columns[col])  # 1 value
        mad_value = mad_mag(mag_columns[col])  # 1 value
        max_value = max_mag(mag_columns[col])  # 1 value
        min_value = min_mag(mag_columns[col])  # 1 value
        sma_value = t_sma_mag(mag_columns[col])  # 1 value
        energy_value = t_energy_mag(mag_columns[col])  # 1 value
        IQR_value = IQR_mag(mag_columns[col])  # 1 value
        entropy_value = entropy_mag(mag_columns[col])  # 1 value
        col_mag_values = [mean_value, std_value, mad_value, max_value, min_value, sma_value, energy_value, IQR_value,
                          entropy_value]
        t_mag_features = t_mag_features+col_mag_values
    return t_mag_features


# --------------------------------------------List of ALL FEATURE names--------------------------------------------
def time_features_names():
    # Generating time feature names
    # time domain axial signals' names
    t_axis_signals = [['t_body_AccX', 't_body_AccY', 't_body_AccZ'],
                      ['t_grav_AccX', 't_grav_AccY', 't_grav_AccZ', ],
                      ['t_body_jerk_AccX', 't_body_jerk_AccY', 't_body_jerk_AccZ', ],
                      ['t_body_GyroYaw', 't_body_GyroPitch', 't_body_GyroRoll', ],
                      ['t_body_jerk_GyroYaw', 't_body_jerk_GyroPitch', 't_body_jerk_GyroRoll']]
    
    # time domain magnitude signals' names
    magnitude_signals = ['t_body_Acc_mag', 't_grav_Acc_mag', 't_body_jerk_Acc_mag', 't_body_Gyro_mag',
                         't_body_jerk_Gyro_mag']
    
    # functions' names:
    t_one_input_features_name1 = ['_mean()', '_std()', '_mad()', '_max()', '_min()']
    t_one_input_features_name2 = ['_energy()', '_iqr()', '_entropy()']
    
    # correlation_columns = ['_Corr(X,Y)', '_Corr(X,Z)', '_Corr(Y,Z)']
    features = []  # Empty list : it will contain all time domain features' names
    
    for columns in t_axis_signals:  # iterate throw  each group of 3-axial signals'
        for feature in t_one_input_features_name1:  # iterate throw the first list of functions names
            for column in columns:  # iterate throw each axial signal in that group
                new_column = assert_middle_feature(column, feature)  # build the feature name
                features.append(new_column)  # add it to the global list
        sma_column = remove_direction_feature(column)+'_sma()'  # build the feature name sma related to that group
        features.append(sma_column)  # add the feature to the list
        
        for feature in t_one_input_features_name2:  # same process for the second list of features functions
            for column in columns:
                new_column = assert_middle_feature(column, feature)
                features.append(new_column)
        
        # for feature in correlation_columns:  # adding correlations features
        #     new_column = remove_direction_feature(column)+feature
        #     features.append(new_column)
    
    for columns in magnitude_signals:  # iterate throw time domain magnitude column names
        # build feature names related to that column
        # list 1
        for feature in t_one_input_features_name1:
            new_column = columns+feature
            features.append(new_column)
        # sma feature name
        sma_column = columns+'_sma()'
        features.append(sma_column)
        
        # list 2
        for feature in t_one_input_features_name2:
            new_column = columns+feature
            features.append(new_column)
    
    time_list_features = features
    
    return time_list_features  # return all time domain features' names


# =========================================================== Functions for FREQUENCY features==========================
# --------------------------------------------List of 3-AXIAL FREQUENCY FEATURES----------------------------------------

# sma
def f_sma_axial(df):
    array = np.array(df)
    sma_value = float((abs(array)/math.sqrt(window_size)).sum())/float(3)  # sma value of 3-axial f_signals
    return sma_value


# energy
def f_energy_axial(df):
    array = np.array(df)
    # spectral energy vector
    energy_vector = list((array ** 2).sum(axis=0)/float(len(array)))  # energy of: f_signalX,f_signalY, f_signalZ
    return energy_vector  # energy vector= [energy(signal_X),energy(signal_Y),energy(signal_Z)]


freqs = fftfreq(window_size, d=1/sampling_freq)


# max_Inds
def f_max_Inds_axial(df):
    array = np.array(df)
    max_Inds_X = freqs[
        array[1:int(window_size/2+1), 0].argmax()+1]  # return the frequency related to max value of f_signal X
    max_Inds_Y = freqs[
        array[1:int(window_size/2+1), 1].argmax()+1]  # return the frequency related to max value of f_signal Y
    max_Inds_Z = freqs[
        array[1:int(window_size/2+1), 2].argmax()+1]  # return the frequency related to max value of f_signal Z
    max_Inds_vector = [max_Inds_X, max_Inds_Y, max_Inds_Z]  # put those frequencies in a list
    return max_Inds_vector


# mean freq()
def f_mean_Freq_axial(df):
    array = np.array(df)
    # sum of( freq_i * f_signal[i])/ sum of signal[i]
    mean_freq_X = np.dot(freqs, array[:, 0]).sum()/float(
            array[:, 0].sum())  # frequencies weighted sum using f_signalX
    mean_freq_Y = np.dot(freqs, array[:, 1]).sum()/float(
            array[:, 1].sum())  # frequencies weighted sum using f_signalY
    mean_freq_Z = np.dot(freqs, array[:, 2]).sum()/float(
            array[:, 2].sum())  # frequencies weighted sum using f_signalZ
    mean_freq_vector = [mean_freq_X, mean_freq_Y, mean_freq_Z]  # vector contain mean frequencies[X,Y,Z]
    return mean_freq_vector


def f_skewness_and_kurtosis_axial(df):
    array = np.array(df)
    skew_X = skew(array[:, 0])  # skewness value of signal X
    kur_X = kurtosis(array[:, 0])  # kurtosis value of signal X
    skew_Y = skew(array[:, 1])  # skewness value of signal Y
    kur_Y = kurtosis(array[:, 1])  # kurtosis value of signal Y
    skew_Z = skew(array[:, 2])  # skewness value of signal Z
    kur_Z = kurtosis(array[:, 2])  # kurtosis value of signal Z
    skew_kur_3axial_vector = [skew_X, kur_X, skew_Y, kur_Y, skew_Z, kur_Z]  # return the list
    return skew_kur_3axial_vector


# --------------------------------------------List of 3-AXIAL FREQUENCY FEATURE generation------------------------------
def f_axial_features_generation(f_window):
    axial_columns = f_window.columns[0:12]  # select frequency axial column names
    axial_df = f_window[axial_columns]  # select frequency axial signals in one dataframe
    f_all_axial_features = []  # a global list will contain all frequency axial features values
    for col in range(0, 12, 3):  # iterate throw each group of frequency axial signals in a window
        df = axial_df[axial_columns[col:col+3]]  # select each group of 3-axial signals
        # mean
        mean_vector = mean_axial(df)  # 3 values
        # std
        std_vector = std_axial(df)  # 3 values
        # mad
        mad_vector = mad_axial(df)  # 3 values
        # max
        max_vector = max_axial(df)  # 3 values
        # min
        min_vector = min_axial(df)  # 3 values
        # sma
        sma_value = f_sma_axial(df)
        # energy
        energy_vector = f_energy_axial(df)  # 3 values
        # IQR
        IQR_vector = IQR_axial(df)  # 3 values
        # entropy
        entropy_vector = entropy_axial(df)  # 3 values
        # max_inds
        max_inds_vector = f_max_Inds_axial(df)  # 3 values
        # skewness and kurtosis
        skewness_and_kurtosis_vector = f_skewness_and_kurtosis_axial(df)  # 6 values
        # append all values of each 3-axial signals in a list
        f_3axial_features = mean_vector+std_vector+mad_vector+max_vector+min_vector+[sma_value]+energy_vector+ \
                            IQR_vector+entropy_vector+max_inds_vector+skewness_and_kurtosis_vector
        f_all_axial_features = f_all_axial_features+f_3axial_features  # add features to the global list
    return f_all_axial_features


# --------------------------------------------List of FREQUENCY MAGNITUDE FEATURE---------------------------------------
# Functions used to generate frequency magnitude features

# sma
def f_sma_mag(mag_column):
    array = np.array(mag_column)
    sma_value = float((abs(array)/math.sqrt(len(mag_column))).sum())  # sma of one mag f_signals
    return sma_value


# energy
def f_energy_mag(mag_column):
    array = np.array(mag_column)
    # spectral energy value
    energy_value = float((array ** 2).sum()/float(len(array)))  # energy value of one mag f_signals
    return energy_value


# max_Inds
def f_max_Inds_mag(mag_column):
    array = np.array(mag_column)
    max_Inds_value = float(
            freqs[array[1:int(window_size/2+1)].argmax()+1])  # freq value related with max component
    return max_Inds_value


# mean freq()
def f_mean_Freq_mag(mag_column):
    array = np.array(mag_column)
    mean_freq_value = float(np.dot(freqs, array).sum()/float(array.sum()))  # weighted sum of one mag f_signal
    return mean_freq_value


# skewness
def f_skewness_mag(mag_column):
    array = np.array(mag_column)
    skew_value = float(skew(array))  # skewness value of one mag f_signal
    return skew_value


# kurtosis
def f_kurtosis_mag(mag_column):
    array = np.array(mag_column)
    kurtosis_value = float(kurtosis(array))  # kurtosis value of on mag f_signal
    return kurtosis_value


# --------------------------------------------List of FREQUENCY MAGNITUDE FEATURE generation----------------------------
def f_mag_features_generation(f_window):
    # select frequency mag columns : the last 4 columns in f_window
    mag_columns = f_window.columns[-4:]
    mag_columns = f_window[mag_columns]
    f_mag_features = []
    for col in mag_columns:  # iterate throw each mag column in f_window
        # calculate common mag features and frequency mag features for each column
        mean_value = mean_mag(mag_columns[col])
        std_value = std_mag(mag_columns[col])
        mad_value = mad_mag(mag_columns[col])
        max_value = max_mag(mag_columns[col])
        min_value = min_mag(mag_columns[col])
        sma_value = f_sma_mag(mag_columns[col])
        energy_value = f_energy_mag(mag_columns[col])
        IQR_value = IQR_mag(mag_columns[col])
        entropy_value = entropy_mag(mag_columns[col])
        max_Inds_value = f_max_Inds_mag(mag_columns[col])
        skewness_value = f_skewness_mag(mag_columns[col])
        kurtosis_value = f_kurtosis_mag(mag_columns[col])
        col_mag_values = [mean_value, std_value, mad_value, max_value, min_value, sma_value, energy_value, IQR_value,
                          entropy_value, max_Inds_value, skewness_value, kurtosis_value]
        f_mag_features = f_mag_features+col_mag_values  # append feature values of one mag column to the global list
    return f_mag_features


# --------------------------------------------List of ALL FREQUENCY FEATURE generation----------------------------------

def frequency_features_names():
    # Generating Frequency feature names
    # frequency axial signal names 
    axial_signals = [['f_body_AccX', 'f_body_AccY', 'f_body_AccZ'],
                     ['f_body_jerk_AccX', 'f_body_jerk_AccY', 'f_body_jerk_AccZ'],
                     ['f_body_GyroYaw', 'f_body_GyroPitch', 'f_body_GyroRoll'],
                     ['f_body_jerk_GyroYaw', 'f_body_jerk_GyroPitch', 'f_body_jerk_GyroRoll'], ]
    
    # frequency magnitude signals
    mag_signals = ['f_body_Acc_mag', 'f_body_jerk_Acc_mag', 'f_body_Gyro_mag', 'f_body_jerk_Gyro_mag']
    
    # features functions names will be applied to f_signals
    f_one_input_features_name1 = ['_mean()', '_std()', '_mad()', '_max()', '_min()']
    f_one_input_features_name2 = ['_energy()', '_iqr()', '_entropy()', '_maxInd()']
    f_one_input_features_name3 = ['_skewness()', '_kurtosis()']
    frequency_features_names = []  # global list of frequency features
    for columns in axial_signals:  # iterate throw each group of 3-axial signals
        # iterate throw the first list of features
        for feature in f_one_input_features_name1:
            for column in columns:  # iterate throw each signal name of that group
                new_column = assert_middle_feature(column, feature)  # build the full feature name
                frequency_features_names.append(new_column)  # add the feature name to the global list
        
        # sma feature name
        sma_column = remove_direction_feature(column)+'_sma()'
        frequency_features_names.append(sma_column)
        
        # iterate throw the first list of features
        for feature in f_one_input_features_name2:
            for column in columns:
                new_column = assert_middle_feature(column, feature)
                frequency_features_names.append(new_column)
        
        # iterate throw each signal name of that group
        for column in columns:
            for feature in f_one_input_features_name3:  # iterate throw [skewness ,kurtosis]
                new_column = assert_middle_feature(column, feature)  # build full feature name
                frequency_features_names.append(new_column)  # append full feature names
    
    # generate frequency mag features names
    for column in mag_signals:  # iterate throw each frequency mag signal name
        for feature in f_one_input_features_name1:  # iterate throw the first list of features functions names
            frequency_features_names.append(
                    column+feature)  # build the full feature name and add it to the global list
        
        sma_column = column+'_sma()'  # build the sma full feature name
        frequency_features_names.append(sma_column)  # add it to the global list
        
        for feature in f_one_input_features_name2:  # iterate throw the second list of features functions names
            frequency_features_names.append(
                    column+feature)  # build the full feature name and add it to the global list
        
        for feature in f_one_input_features_name3:  # iterate throw the third list of features functions names
            frequency_features_names.append(
                    column+feature)  # build the full feature name and add it to the global list
    
    return frequency_features_names


# =========================================================== Functions for ADDITIONAL features=========================
def magnitude_vector(vector3D):  # vector[X,Y,Z]
    return sqrt((vector3D ** 2).sum())  # euclidian norm of that vector


def angle(vector1, vector2):
    vector1_mag = magnitude_vector(vector1)  # euclidian norm of V1
    vector2_mag = magnitude_vector(vector2)  # euclidian norm of V2
    
    scalar_product = np.dot(vector1, vector2)  # scalar product of vector 1 and Vector 2
    cos_angle = scalar_product/float(vector1_mag*vector2_mag)  # the cosin value of the angle between V1 and V2
    
    # just in case some values were added automatically
    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1
    
    angle_value = float(acos(cos_angle))  # the angle value in radian
    return angle_value  # in radian.


def angle_features(t_window):  # it returns 7 angles per window
    angles_list = []  # global list of angles values
    
    # mean value of each column t_body_acc[X,Y,Z]
    V2_columns = ['t_grav_AccX', 't_grav_AccY', 't_grav_AccZ']
    V2_Vector = np.array(t_window[V2_columns].mean())  # mean values
    
    # angle 0: angle between (t_body_acc[X.mean,Y.mean,Z.mean], t_gravity[X.mean,Y.mean,Z.mean])
    V1_columns = ['t_body_AccX', 't_body_AccY', 't_body_AccZ']
    V1_Vector = np.array(t_window[V1_columns].mean())  # mean values of t_body_acc[X,Y,Z]
    angles_list.append(angle(V1_Vector, V2_Vector))  # angle between the vectors added to the global list
    
    # same process is applied to other signals
    # angle 1: (t_body_acc_jerk[X.mean,Y.mean,Z.mean],t_gravity[X.mean,Y.mean,Z.mean]
    V1_columns = ['t_body_jerk_AccX', 't_body_jerk_AccY', 't_body_jerk_AccZ']
    V1_Vector = np.array(t_window[V1_columns].mean())
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # angle 2: (t_body_gyro[X.mean,Y.mean,Z.mean],t_gravity[X.mean,Y.mean,Z.mean]
    V1_columns = ['t_body_GyroYaw', 't_body_GyroPitch', 't_body_GyroRoll']
    V1_Vector = np.array(t_window[V1_columns].mean())
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # angle 3: (t_body_gyro_jerk[X.mean,Y.mean,Z.mean],t_gravity[X.mean,Y.mean,Z.mean]
    V1_columns = ['t_body_jerk_GyroYaw', 't_body_jerk_GyroPitch', 't_body_jerk_GyroRoll']
    V1_Vector = np.array(t_window[V1_columns].mean())
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # V1 vector in this case is the X axis itself [1,0,0]
    # angle 4: ([X_axis],t_gravity[X.mean,Y.mean,Z.mean])   
    V1_Vector = np.array([1, 0, 0])
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # V1 vector in this case is the Y axis itself [0,1,0]
    # angle 5: ([Y_acc_axis],t_gravity[X.mean,Y.mean,Z.mean]) 
    V1_Vector = np.array([0, 1, 0])
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # V1 vector in this case is the Z axis itself [0,0,1]
    # angle 6: ([Z_acc_axis],t_gravity[X.mean,Y.mean,Z.mean])
    V1_Vector = np.array([0, 0, 1])
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    return angles_list


angle_columns = ['angle0()', 'angle1()', 'angle2()', 'angle3()', 'angle4()', 'angle5()', 'angle6()']


# =========================================================== Functions for ADDITIONAL features=========================

def create_window_df(dic_1, dic_2):
    data_dic = dict()
    for key in dic_1.keys():
        df_1 = dic_1["t_"+key[2:]]
        df_2 = dic_2["f_"+key[2:]]
        df = pd.concat([df_1, df_2], axis=1)
        data_dic[key[2:]] = df
    return data_dic


def Dataset_Generation_PipeLine(t_dic, f_dic):
    """Dataset Generation pipeline
    Args: 
        t_dic is a dic contains time domain windows
        f_dic is a dic contains frequency domain windows
        f_dic should be the result of applying fft to t_dic
    Returns: 
        Composed dataframe
    """
    # concatenate all features names lists and we add two other columns activity ids and user ids will be related to
    # each row
    all_columns = time_features_names()+frequency_features_names()+angle_columns+['activity_Id', 'user_Id']
    
    logger.debug(f"length of each feature component in dataset: all_columns={len(all_columns)}, time_features_names={len(time_features_names())}, frequency_features_names={len(frequency_features_names())}, angle_columns={len(angle_columns)}, labels=2")
    
    final_Dataset = pd.DataFrame(data=[], columns=all_columns)  # build an empty dataframe to append rows
    
    assert set([i[2:] for i in t_dic.keys()]) == set([i[2:] for i in f_dic.keys()])
    
    for i in range(len(t_dic)):  # iterate throw each window
        # t_window and f_window should have the same window id included in their keys
        t_key = sorted(t_dic.keys())[i]  # extract the key of t_window
        f_key = sorted(f_dic.keys())[i]  # extract the key of f_window
        
        t_window = t_dic[t_key]  # extract the t_window
        f_window = f_dic[f_key]  # extract the f_window
        
        window_user_id = extract_info_window_key(t_key, "user")  # extract the user id from window's key
        window_activity_id = extract_info_window_key(t_key, "activity")  # extract the activity id from the windows key
        
        # generate all time features from t_window 
        time_features = t_axial_features_generation(t_window)+t_mag_features_generation(t_window)
        
        # generate all frequency features from f_window
        frequency_features = f_axial_features_generation(f_window)+f_mag_features_generation(f_window)
        
        # Generate additional features from t_window
        additional_features = angle_features(t_window)
        
        # concatenate all features and append the activity id and the user id
        row = time_features+frequency_features+additional_features+[int(window_activity_id), window_user_id]
        
        # go to the first free index in the dataframe
        free_index = len(final_Dataset)
        
        # append the row
        final_Dataset.loc[free_index] = row
    
    return final_Dataset  # return the final dataset
