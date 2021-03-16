import json
import os
import ast
import math
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from logging import getLogger
from scipy.signal import medfilt, butter, filtfilt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

# ====================================================== UTILITY functions =============================================
def remove_outlier(data_frame, column_gyro, threshold): 
        for i in column_gyro: 
            data_frame = data_frame[(data_frame[i] <= threshold) & (data_frame[i] >= -threshold)]
        return data_frame
    

    
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


def getSignedInt(number, bitSize):
    msb = number >> (bitSize-1)
    if msb:  # is negative due to msb being 1, do inverse 2's complement
        int_max = 2 ** bitSize-1
        complement = int_max+1-number
        complement = complement*-1
        return complement
    else:
        return number


def decodeDataField(field):
    flippedData = flipDataBytes(field)
    return getSignedInt(flippedData, 16)


def flipDataBytes(dataBytes):
    firstByte = (dataBytes & 0xFF00) >> 8
    secondByte = (dataBytes & 0xFF) << 8
    return firstByte+secondByte

def change_unit(df): 
    for i in df.columns: 
        if "Acc" in i: 
            df[i] = df[i]/(2**14) # to get 1g
        elif "Gyro" in i: 
            df[i] = df[i]/180.0*math.pi # to get rad/s
    return df


def combine_all_dfs(raw_dic, column_names): 
    data_frames = [df[column_names] for df in raw_dic.values()]
    data_frames = pd.concat(data_frames).reset_index(drop=True)
    return data_frames


def single_scale(signal, user_key, column_names=[], scaler="standard", minmax_range=(0, 1)):
    if scaler == "standard":
        sc = StandardScaler()
        signal = sc.fit_transform(signal)
        
    elif scaler == "minmax":
        sc = MinMaxScaler(feature_range=minmax_range)
        signal = sc.fit_transform(signal)
    
    column_agg = "_".join(column_names)
    utils.pickle_save(f'{DATA_DIR}/ProcessedData/scaler_{user_key}[{scaler}][{column_agg}].pkl', sc)
    utils.pickle_save(f'{TEST_DIR}/Scaler/scaler_{user_key}[{scaler}][{column_agg}].pkl', sc)
    return pd.DataFrame(signal, columns=column_names)
    
    
def create_scaler(raw_dic, column_names, scaler="standard"): 
    for key in raw_dic.keys(): 
        df = raw_dic[key].copy()
        single_scale(df[column_names], key, column_names, scaler=scaler)

        
def scale(signal, column_names, path):
    sc = utils.pickle_load(path)
    scaled_signal = sc.transform(signal)
    return pd.DataFrame(scaled_signal, columns=column_names)


def preprocess(data_frame, column_names, key, postfix="sub"): 
    for i in column_names: 
        data_frame[i] = data_frame[i].diff(1)
    data_frame.dropna(how='any', inplace=True)        
    data_frame.dropna(how='any', inplace=True)
    data_frame["experiment_number_ID"] = f"{key}{postfix}"
    return data_frame


def write_csv(data_dic, path):
    for key in data_dic.keys(): 
        data_dic[key].to_csv(f"{path}/{key}.csv")
    
# ============================================================ Main importing functions ================================
def import_raw_signals_v1(file_path):
    """Read the old text file format (collect during week 4)
    Args: 
        file_path: location of file
        columns: the intended list of columns name
    Returns: 
        A dataframe object
    """
    opened_file = list(open(file_path, 'r', encoding="utf8"))
    raw_data_opened_file = [i for i in opened_file if
                            i.startswith("{TIMEPLOT|DATA|Filtered-") and any(map(str.isdigit, i))]
    index_retrieved_end = int(len(raw_data_opened_file)/6)*6
    opened_file_dict = dict()
    for line in raw_data_opened_file[:index_retrieved_end]:
        line = line.replace("{TIMEPLOT|DATA|Filtered-", "").replace("}", "").replace("|T|", "|").replace("\n", "")
        data = line.split("|")
        if data[0] not in opened_file_dict.keys():
            opened_file_dict[data[0]] = []
        if len(data) > 1:
            opened_file_dict[data[0]].append(float(data[1]))
    data_frame = pd.DataFrame.from_dict(opened_file_dict)
    return data_frame


def import_raw_signals_v2(file_path, columns=['AccX', 'AccY', 'AccZ', 'GyroYaw', 'GyroPitch', 'GyroRoll']):
    """Read the text file format (collect during week 6 and partially week 8)
    Args: 
        file_path: location of file
        columns: the intended list of columns name
    Returns: 
        A dataframe object
    """
    opened_file_dict = dict()
    for i in columns:
        opened_file_dict[i] = []
    
    opened_file = list(open(file_path, 'r', encoding="utf8"))
    raw_data_opened_file = [i for i in opened_file if i.startswith("command packet:") and any(map(str.isdigit, i))]
    index_retrieved_end = int(len(raw_data_opened_file)/6)*6
    
    for data in raw_data_opened_file[:index_retrieved_end]:
        data = data.replace("command packet:", "")
        data = int(data[2:], 16)
        opened_file_dict[columns[0]].append(decodeDataField((data & 0xFFFF000000000000000000000000) >> 96))
        opened_file_dict[columns[1]].append(decodeDataField((data & 0xFFFF00000000000000000000) >> 80))
        opened_file_dict[columns[2]].append(decodeDataField((data & 0xFFFF0000000000000000) >> 64))
        opened_file_dict[columns[3]].append(decodeDataField((data & 0xFFFF000000000000) >> 48))
        opened_file_dict[columns[4]].append(decodeDataField((data & 0xFFFF00000000) >> 32))
        opened_file_dict[columns[5]].append(decodeDataField((data & 0xFFFF0000) >> 16))
    data_frame = pd.DataFrame.from_dict(opened_file_dict)
    return data_frame

def import_raw_signals_v3(file_path, contents=None, columns=['AccX', 'AccY', 'AccZ', 'GyroYaw', 'GyroPitch', 'GyroRoll']):
    """Read the text file format (collect during week 8)
    Args: 
        file_path: location of file
        columns: the intended list of columns name
    Returns: 
        A dataframe object
    """
    
    opened_file_dict = dict()
    contents = list(open(file_path, 'r', encoding="utf8")) if contents is None else contents
    if not contents[0].startswith("{"): 
        columns = columns + ["RealTime"]
    
    for i in columns:
        opened_file_dict[i] = []
    
    for i in contents: 
        if not i.startswith("{"): 
            opened_file_dict[columns[6]].append(datetime.strptime(i.split("{")[0], "%Y-%m-%d %H:%M:%S.%f: ").strftime("%Y-%m-%d %H:%M:%S.%f"))
            i = "{" + i.split("{")[1]
        dictionary = ast.literal_eval(i)
        opened_file_dict[columns[0]].append(dictionary["xAccel"])
        opened_file_dict[columns[1]].append(dictionary["yAccel"])
        opened_file_dict[columns[2]].append(dictionary["zAccel"])
        opened_file_dict[columns[3]].append(dictionary["yaw"])
        opened_file_dict[columns[4]].append(dictionary["pitch"])
        opened_file_dict[columns[5]].append(dictionary["row"])
    data_frame = pd.DataFrame.from_dict(opened_file_dict)
    return data_frame

def create_raw_dic(
    Raw_data_paths, Activities_name_index, Member_name_index, Experiment_ids,
    scaler_name=f"{DATA_DIR}/ProcessedData/scaler_Chi_wk8(2)[standard][AccX_AccY_AccZ_GyroYaw_GyroPitch_GyroRoll]",
    raw_columns=['AccX', 'AccY', 'AccZ', 'GyroYaw', 'GyroPitch', 'GyroRoll'], 
    mode="general"):
    
    """Create a dictionary where all dataframes will be stored
    Args: 
        raw_columns: List of column names in the csv file
        Raw_data_paths: List of file names to be extracted
        Activities_name_index: dictionary of activities name
        Member_name_index: dictionary of user name
    Returns:
        a dictionary where all dataframes will be stored
    """
    column_names = ['AccX', 'AccY', 'AccZ', 'GyroYaw', 'GyroPitch', 'GyroRoll']
    column_gyro  = ['GyroYaw', 'GyroPitch', 'GyroRoll']

    raw_dic = {}
    raw_signals_data_frame = pd.DataFrame()
    flag = True
    
    for file in Raw_data_paths:
        user = experiment_key = ""
        head, tail = os.path.split(file)
        file_name = tail.replace(".txt", "")
        keys = file_name.split("-")

        if len(keys) == 3:
            user = keys[1]
            activitity_key = keys[0].upper()
            experiment_key = keys[2]
            raw_signals_data_frame = import_raw_signals_v1(file)
            raw_signals_data_frame['activity_ID'] = activitity_key
            raw_signals_data_frame['activity_number_ID'] = Activities_name_index[activitity_key]
        
        elif len(keys) == 2:
            user = keys[0]
            experiment_key = keys[1]
            if experiment_key in Experiment_ids: 
                raw_signals_data_frame = import_raw_signals_v2(file, columns=raw_columns)
            else: 
                raw_signals_data_frame = import_raw_signals_v3(file, columns=raw_columns)
                
        user_key = f"{user}_{experiment_key}"
        if user_key not in raw_dic.keys():
            raw_dic[user_key] = pd.DataFrame()
            
        raw_signals_data_frame['experiment_number_ID'] = experiment_key
        raw_signals_data_frame['user_number_ID'] = Member_name_index[user]
        data_frame = pd.concat([raw_dic[user_key], raw_signals_data_frame], axis=0, ignore_index=True)        
        data_frame["time"] = 1/float(sampling_freq)*data_frame.index
        data_frame = remove_outlier(data_frame, column_gyro, 180)
        raw_dic[user_key] = data_frame
    
    raw_dic_new = dict() if mode=="streaming" else raw_dic.copy()
    
    create_scaler(raw_dic, column_names, scaler="standard")
    create_scaler(raw_dic, column_names, scaler="minmax")
    
    scaler_path = f"{scaler_name}.pkl"
    post_fix = "(scSub)"
    
    for key in raw_dic.keys(): 
        df = raw_dic[key].copy()
        df[column_names] = scale(df[column_names], column_names, scaler_path)
        df.dropna(how='any', inplace=True)
        df = preprocess(df, column_names, key, postfix=post_fix)
        raw_dic_new[f"{key}{post_fix}"] = df
        df["experiment_number_ID"] = f"{key}{post_fix}"
    
    for key in raw_dic_new.keys(): 
        df = apply_filter(raw_dic_new[key].copy(), filter="mean")
        df = df[column_names]
        raw_dic_new[key] = df
    
    
    if mode=="streaming": 
        print("Entering Streaming Mode")
        return {list(raw_dic_new.keys())[-1]: raw_dic_new[list(raw_dic_new.keys())[-1]].tail(window_size)}, list(raw_dic_new.keys())[-1]
    elif mode=="file": 
        print("Entering File Mode")
        return {list(raw_dic_new.keys())[-1]: raw_dic_new[list(raw_dic_new.keys())[-1]]}, [post_fix], list(raw_dic_new.keys())[-1]
    else: 
        return raw_dic_new, [post_fix]


def get_labels_data_frame(path, Activities_name_index, Member_name_index, postfix_list, raw_dic=None):
    """Creating a dataframe where labels dataframes will be stored
    Args: 
        path: path of the file
        raw_dic: raw data dictionaries
        Activities_name_index: dictionary of activities name
        Member_name_index: dictionary of user name
    Returns: 
        Labels dataframe
    """
    final_labels_columns = ["experiment_number_ID", "user_number_ID", "user_ID", "activity_number_ID",
                            "activity_ID", "Label_start_point", "Label_end_point"]
    df = pd.read_csv(path)
    
    df["activity_ID"] = df["activity_ID"].apply(lambda x: x.strip().upper())
    df["activity_number_ID"] = df["activity_ID"].apply(lambda x: Activities_name_index[x])
    df["user_number_ID"] = df["user_ID"].apply(lambda x: Member_name_index[x])
    df["dic_key"] = df["user_ID"]+"_"+df["experiment_number_ID"]
    
    if raw_dic: 
        df = df[df["dic_key"].isin(raw_dic.keys())].dropna().astype({"Label_start_point": int, "Label_end_point": int})
    
    print("SET: ", set(df["dic_key"]))
    
    df_new = df.copy()
    for i in postfix_list: 
        df_tmp = df_new.copy()
        df_tmp["experiment_number_ID"] = df["experiment_number_ID"]+f"{i}"
        df_new = pd.concat([df_new, df_tmp])
    
    df = df_new.copy()
    df["diff"] = df["Label_end_point"]-df["Label_start_point"]
    
    df = df[final_labels_columns]
    df.sort_values(by=['experiment_number_ID', 'user_ID', 'Label_start_point'], inplace=True)
    df.reset_index(inplace=True)
    df = df[final_labels_columns]
    
    df["key"] = df["user_ID"]+"_"+df["experiment_number_ID"]
    
    logger.debug('Description of Labels dataframe \n\n\t'+df.describe().to_string().replace('\n', '\n\t'))
    return df
