import json
from logging import getLogger

import numpy as np
import pandas as pd

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


def find_activity(Labels_Data_Frame, key, cursor, end_point):
    """Voting function used to find the major activity in the window
    Args: 
        Labels_Data_Frame: Dataframe defined earlier
        key: user key
        cursor: index of the first row of a window
        end_point: point ending window
    Returns: 
        activity
    """
    exp_AL = np.array(Labels_Data_Frame[Labels_Data_Frame['key'] == key])
    
    # the near starting point above(inferior) the cursor
    St_p1 = exp_AL[exp_AL[:, 5] <= cursor][:, 5].max()  # starting point label of the first activity
    
    # the near ending point below the cursor (superior ) 
    En_p2 = exp_AL[exp_AL[:, 6] >= end_point][:, 6].min()  # ending point label of the second activity
    
    
    for index in range(len(exp_AL)):  # iterating over each line
        if exp_AL[index, 5] == St_p1:  # selecting the row index with starting point
            Ind1 = index  # index of the first activity
        if exp_AL[index, 6] == En_p2:  # selecting the row index with ending point ==b
            Ind2 = index  # the index of the second activity
    if Ind1 == Ind2:  # if the window rows indexes are inside the same activity data points which means
        # that the first and second activity are actually the same
        activity = exp_AL[Ind1, 3]  # window_activity id will take the same value as the activity_label of these rows
    else:
        # exp_AL[Ind1,6] is the ending_point_label of the first activity
        if cursor+int(window_size/2)-1 <= exp_AL[Ind1, 6]:  # if the first int(window_size/2) data points or more
            # are included in the first activity bands
            activity = exp_AL[Ind1, 3]  # the window will take the same activity labels as the first activity
        # exp_AL[Ind2,5] is the starting_point_label of the second activity
        elif cursor+int(window_size/2) >= exp_AL[Ind2, 5]:  # if the last int(window_size/2) data points or more
            # are included in the second activity bands
            activity = exp_AL[Ind2, 3]  # the window will take the activity labels as the second activity
        else:  # if  more than int(window_size/2) data point doesn't belong (all at once) neither to first activity
            # or to the second activity
            return None  # this window activity label will be equal to None
    return activity  # to convert to string and add a '0' to the left if activity <10


def Windowing_type_1(time_sig_dic, columns, Labels_Data_Frame=None):
    """Creating window-based dictionary of dataframes, window strictly contains only 1 type of activity
    Args: 
        time_sig_dic(dict): time-based dictionary includes all time domain signals' dataframes
        columns: raw data dictionaries columns
        Labels_Data_Frame: labels file defined earlier
    Returns: 
        window-based dictionary of dataframes
    """
    window_ID = 0  # window unique id
    t_dic_win_type_I = {}  # output dic
    BA_array = np.array(Labels_Data_Frame[(Labels_Data_Frame["activity_number_ID"] < 11)])  # Just Basic activities
    for line in BA_array:
        # extracting the dataframe key that contains rows related to this activity [expID,userID]
        file_key = f"{line[2]}_{line[0]}"
        # extract the activity id in this line
        act_ID = line[3]
        # starting point index of an activity
        start_point = int(line[5])
        # from the cursor we copy a window that has window_size rows
        for cursor in range(start_point, int(line[6])-int(window_size)-1):
            # end_point: cursor(the first index in the window) + window_size
            end_point = cursor+window_size  # window end row
            # selecting window data points convert them to numpy array to delete rows index
            data = np.array(time_sig_dic[file_key].iloc[cursor:end_point])
            # converting numpy array to a dataframe with the same column names
            window = pd.DataFrame(data=data, columns=columns)
            # creating the window
            key = 't_W'+utils.normalize5(window_ID)+'_'+file_key+'_act'+utils.normalize2(act_ID)
            t_dic_win_type_I[key] = window
            # incrementing the windowID by 1
            window_ID = window_ID+1
    return t_dic_win_type_I  # return a dictionary including time domain windows type I


def Windowing_type_2(time_sig_dic, columns, Labels_Data_Frame=None):
    """Creating window-based dictionary of dataframes, window can contains > 1 type of activity
    Args: 
        time_sig_dic(dict): time-based dictionary includes all time domain signals' dataframes
        Labels_Data_Frame: label df
        columns: raw data dictionaries columns
    Returns: 
        window-based dictionary of dataframes
    """
    window_ID = 0
    t_dic_win_type_II = {}
    
    for key, data_frame in time_sig_dic.items(): 
        exp_array = np.array(data_frame)
        
        exp_AL_array = np.array(Labels_Data_Frame[Labels_Data_Frame['key'] == key])
        
        if exp_AL_array.shape[0] > 0: 
            # The first starting_point_label in this experiment
            start_point = exp_AL_array[0, 5]

            # The last ending_point_label of this experiment
            end_point = exp_AL_array[-1, 6]

            for cursor in range(start_point, end_point-window_size-1): # cursor represents index of first data point in window
                end_window = cursor+window_size-1  # end_window is the index of the last data point in a window
                # creating the window
                window_array = exp_array[cursor:end_window+1, :]
                
                # Determining the appropriate activity label of this window
                act_ID = find_activity(Labels_Data_Frame, key, cursor, end_window)
                
                if act_ID is not None:  # if act_ID is none  this window doesn't belong to any activity
                    # since the act_ID is != to None the window array will be stored in DataFrame with the appropriate
                    # column names
                    window = pd.DataFrame(data=window_array, columns=columns)
                
                    # Generating the window key(unique key :since the window_ID is unique for each window )
                    # I chose to add the exp, user, activity Identifiers in the win_key they will be useful later. (
                    # exp: optional)
                    win_key = 't_W'+utils.normalize5(window_ID)+'_'+key+'_act'+utils.normalize2(act_ID)  
                    # eg: 'W_00000_exp01_user01_act01'
                    if len(window.index) == window_size:
                        # Store the window data frame in a dic
                        t_dic_win_type_II[win_key] = window
                    # Incrementing window_ID by 1
                    window_ID = window_ID+1
    return t_dic_win_type_II

def Windowing_type_3(time_sig_dic, columns, Labels_Data_Frame=None, stride=0):
    """Creating window-based dictionary of dataframes, based on streaming activity
    Args: 
        time_sig_dic(dict): time-based dictionary includes all time domain signals' dataframes
        Labels_Data_Frame: label df
        columns: raw data dictionaries columns
    Returns: 
        window-based dictionary of dataframes
    """
    stride = int(window_size/2) if stride == 0 else stride
    index_list = []
    act_ID_list = []
    window_ID = 0
    t_dic_win_type_III = {}
    
    
    for key, data_frame in time_sig_dic.items(): 
        exp_array = np.array(data_frame)
        
        exp_AL_array = np.array(Labels_Data_Frame[Labels_Data_Frame['key'] == key])
        start_point = 0
        end_point = exp_AL_array[-1, 6]

        for cursor in range(start_point, end_point-window_size-1, stride): # cursor represents index of first data point in window
            end_window = cursor+window_size-1
            window_df  = data_frame.iloc[cursor:end_window+1, :]
            
            act_ID = find_activity(Labels_Data_Frame, key, cursor, end_window)
            act_ID = -1 if act_ID is None else act_ID
            if act_ID == -1: 
                print(key, cursor, end_window)
            
            win_key = 't_W'+utils.normalize5(window_ID)+'_'+key+'_act'+utils.normalize2(-1)  
            t_dic_win_type_III[win_key] = window_df
            window_ID = window_ID+1

            index_list.append((cursor, end_window))
            act_ID_list.append(act_ID)
    return t_dic_win_type_III, index_list, act_ID_list
    
    