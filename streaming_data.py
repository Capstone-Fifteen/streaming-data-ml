import os
import csv
import scipy
import math
import re
import json
import ast
import logging
import joblib
import operator
import sched, time
import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob
from IPython.display import display
from os.path import join as osp
from timeit import default_timer as timer
from logging import basicConfig, getLogger, StreamHandler, DEBUG, WARNING

import warnings
warnings.filterwarnings('ignore')

from matplotlib.backends.backend_pdf import PdfPages

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
from tensorflow import keras

from datetime import datetime
from pathlib import Path

import src.utils as utils
import src.load as load
import src.segmentation as segment

from collections import Counter

CUR_DIR    = utils.get_full_path_directory(mode="main")
DATA_DIR   = utils.get_full_path_directory(mode="data")
CONFIG_DIR = utils.get_full_path_directory(mode="configs")
LOGS_DIR   = utils.get_full_path_directory(mode="logs")
TEST_DIR   = utils.get_full_path_directory(mode="tests")

with open(f"{CONFIG_DIR}/default.json", "r") as f:
    params        = json.load(f)["meta_params"]
    sampling_freq = params["sampling_freq"]
    window_size   = params["window_size"]
    prefix        = params["prefix"]
    n_splits      = params["n_splits"]
print("Configuration for meta_variables:", params)

sep = "========================================================"
sm = "_______________"
Activities_index_name = pd.read_table(f"{CONFIG_DIR}/activity_labels.txt", sep=" ", header=None).to_dict()[0]
Activities_name_index = dict((y,x) for x,y in Activities_index_name.items())
Member_index_name     = pd.read_table(f"{CONFIG_DIR}/user_labels.txt", sep=" ", header=None).to_dict()[0]
Member_name_index     = dict((y,x) for x,y in Member_index_name.items())
Experiment_ids        = ["wk6", "wk8"]

def define_best_model_path(dataset_type): 
    data_paths = sorted(glob(f"{TEST_DIR}/BestModel/*[_dataset_{dataset_type}_]*"))
    return data_paths

def get_sum_dicts(list_dict): 
    dict_result = dict()
    for dict_ in list_dict: 
        dict_result = dict(Counter(dict_result)+Counter(dict_))
    return dict_result

def print_dict(dct):
    meta_strings = []
    for item, amount in dct.items():  # dct.iteritems() in Python 2
        meta_strings.append("{} ({})".format(item, amount))
    return "\n".join(meta_strings)

def get_models_dict(dataset_type_list): 
    model_dict = dict()
    for num in dataset_type_list: 
        model_names = define_best_model_path(num)        
        for i in dataset_type_list: 
            model_dict[i] = dict()
            for model_name in model_names: 
                abs_model_name = os.path.split(model_name)[1]
                if "dataset_"+str(i) in model_name: 
                    model = keras.models.load_model(model_name)
                    model_dict[i][abs_model_name] = model
    return model_dict

def model_list_predict(model_dict, dataset_type, X_train, writer, logger, logger_summary,
                       start_index=-1, end_index=-1, act_ID=-1, mode="streaming"): 
    meta_string=""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    dict_list = []
    if not model_dict[dataset_type]: 
        print("There is no model fitted for", dataset_type)
        return
    for model_name, model in model_dict[dataset_type].items(): 
        pred = model.predict(X_train)
        if np.isnan(pred.any()): 
            print(f"We have nan for start_index={start_index} and end_index={end_index}:", pred)
        else: 
            model_name = model_name.replace("trained_model_dataset_", "")
            max_pred = pred.argmax(axis=1)
            pred_dict = pd.DataFrame(
                pred, columns=[f"CLASS {i}[{Activities_index_name[i]}]" for i in range(10)]).to_dict('records')[0]
            dict_list.append(pred_dict)
            most_likely_act = f"CLASS {max_pred[0]}[{Activities_index_name[max_pred[0]]}]"
            highest_prob = pred_dict[most_likely_act]

            pred_model_name = f"\n\n{sm}{model_name} is used:{sm}\n"
            pred_final_result = f"{sm}{sm}{max_pred[0]}[{Activities_index_name[max_pred[0]]}] with prob={highest_prob}\n"
            result = pred_model_name + pred_final_result + print_dict(pred_dict)

            meta_string = meta_string + result
            writer.writerow({
                "start_index": start_index, "end_index": end_index, "model": model_name, 
                "activity": most_likely_act, "prob": highest_prob, "real_activity": act_ID})
    
    summary_dict = get_sum_dicts(dict_list)
    summary_dict = {k: v/len(dict_list) for k, v in summary_dict.items()}
    most_likely_act = max(summary_dict, key=summary_dict.get)
    
    pred_model_name_summary = f"\n\n{sm}SUMMARY_TABLE_ACTIVITY_PROPORTION{sm}\n"
    pred_final_result_summary = f"{sm}{sm}Most Likely Activity: {most_likely_act}\n"
    
    meta_string = meta_string + pred_model_name_summary + pred_final_result_summary + print_dict(summary_dict)
    writer.writerow({
        "start_index": start_index, "end_index": end_index, "model": "average", 
        "activity": most_likely_act, "prob": summary_dict[most_likely_act], "real_activity": act_ID})

    if mode=="streaming": 
        logger.debug(meta_string)
        logger_summary.debug(f"{start_index}-{end_index}: Most Likely Activity: {most_likely_act}")
        # print(f"{start_index}-{end_index}: Most Likely Activity: {most_likely_act}")

model_dict  = get_models_dict([6, 3])

# If you have missed the error "The reset parameter is False but there is no n_features_in_ attribute. Is this estimator fitted?"
# !pip install scikit-learn==0.22.2.post1

def predict_streaming(scaler_name): 
    Raw_data_paths = sorted(glob(f"{TEST_DIR}/*.txt")) # assuming only contains 1 text file
    Dataset_type_III, file_name = load.create_raw_dic(
        Raw_data_paths, Activities_name_index, Member_name_index, Experiment_ids, 
        scaler_name=scaler_name, 
        mode="streaming"
    )

    LOG_FILE_DIR = f"{TEST_DIR}/Predictions/predictions_log_{file_name}_streaming"
    LOG_FILE_NAME = f"{LOG_FILE_DIR}.log"
    logger = utils.setup_logger(LOG_FILE_DIR, LOG_FILE_NAME, mode="a")

    LOG_FILE_DIR_summary = f"{TEST_DIR}/Predictions/predictions_csv_{file_name}_streaming"
    LOG_FILE_NAME_summary = f"{LOG_FILE_DIR_summary}.csv"
    logger_summary = utils.setup_logger(LOG_FILE_DIR_summary, LOG_FILE_NAME_summary, mode="a", style="csv_file")
    
    column_names=["start_index", "end_index", "model", "activity", "prob", "real_activity"]
    file = open(f"{TEST_DIR}/Predictions/predictions_summary.csv", "a", newline="")
    writer = csv.DictWriter(file, fieldnames=column_names)

    print(Dataset_type_III)
    index_list = Dataset_type_III[list(Dataset_type_III.keys())[-1]].index
    X_train_III = np.array(list(map(lambda x: x.to_numpy(), list(Dataset_type_III.values()))))
    X_train_III = X_train_III.reshape(
        X_train_III.shape[0], X_train_III.shape[1], X_train_III.shape[2], 1)

    print(X_train_III.shape)
    model_list_predict(model_dict, 3, X_train_III, writer, logger, logger_summary, 
                       start_index=index_list[0], end_index=index_list[1])
    
    file.close()
    
def predict_file(scaler_name, file_name = ""): 
    
    Raw_data_paths = sorted(glob(f"{TEST_DIR}/*.txt")) if file_name=="" else f"{TEST_DIR}/{file_name}.txt"

    Dataset_type_III, postfix_list, file_name = load.create_raw_dic(
        Raw_data_paths, Activities_name_index, Member_name_index, Experiment_ids, 
        scaler_name=scaler_name, 
        mode="file"
    )
    
    LOG_FILE_DIR = f"{TEST_DIR}/Predictions/predictions_log_{file_name}"
    LOG_FILE_NAME = f"{LOG_FILE_DIR}.log"
    logger = utils.setup_logger(LOG_FILE_DIR, LOG_FILE_NAME)
    logger.debug(f"Configuration for meta_variables: {params}")

    LOG_FILE_DIR_summary = f"{TEST_DIR}/Predictions/predictions_csv_{file_name}"
    LOG_FILE_NAME_summary = f"{LOG_FILE_DIR_summary}.csv"
    logger_summary = utils.setup_logger(LOG_FILE_DIR_summary, LOG_FILE_NAME_summary, style="csv_file")
    logger_summary.debug(f"Configuration for meta_variables: {params}")
    logger.debug(f"{sep}Starting to predict streaming data{sep}")

    Labels_Data_Frame = load.get_labels_data_frame(
        f"{CONFIG_DIR}/Labels.csv", Activities_name_index, Member_name_index, postfix_list=postfix_list, raw_dic=None)

    column_names = ["AccX", "AccY", "AccZ", "GyroYaw", "GyroPitch", "GyroRoll"]
    Dataset_type_III_seg, index_list, act_ID_list = segment.Windowing_type_3(
        Dataset_type_III,column_names,Labels_Data_Frame, stride=15)

    X_train_III = np.array(list(map(lambda x: x.to_numpy(), list(Dataset_type_III_seg.values()))))
    X_train_III = X_train_III.reshape(X_train_III.shape[0], X_train_III.shape[1], X_train_III.shape[2], 1)

    column_names=["start_index", "end_index", "model", "activity", "prob", "real_activity"]
    file = open(f"{TEST_DIR}/Predictions/predictions_summary_{file_name}.csv", "a", newline="")
    writer = csv.DictWriter(file, fieldnames=column_names)
    writer.writeheader()
    
    logger.debug(f"{sep}Starting to STREAMING{sep}")
    start_time=timer()
    for idx in tqdm(range(X_train_III.shape[0])): 
        start_index, end_index = index_list[idx]
        act_ID = act_ID_list[idx]
        X_train_III_elem = X_train_III[idx]
        X_train_III_elem = X_train_III_elem.reshape(
            1, X_train_III_elem.shape[0], X_train_III_elem.shape[1], X_train_III_elem.shape[2])
        
        model_list_predict(model_dict, 3, X_train_III_elem, writer, logger, logger_summary, 
                           start_index=start_index, end_index=end_index, act_ID=act_ID, mode="streaming")
    
    end_time=timer()
    logger.debug(f'Duration in seconds : {end_time-start_time}')
    logger.debug(f"{sep}Ending process STREAMING{sep}")
    
    file.close()
    
    return f"{TEST_DIR}/Predictions/predictions_summary_{file_name}.csv"

scaler_name = f"{TEST_DIR}/Scaler/scaler_XH_wk8(1)[standard][AccX_AccY_AccZ_GyroYaw_GyroPitch_GyroRoll]"

tf_logger = getLogger("tensorflow")
tf_logger.setLevel(logging.WARNING)

# (window_size/2)/(stride)*240 = # records

column_names=["start_index", "end_index", "model", "activity", "prob", "real_activity"]
file = open(f"{TEST_DIR}/Predictions/predictions_summary.csv", "a", newline="")
writer = csv.DictWriter(file, fieldnames=column_names)
writer.writeheader()

while(1):
    predict_streaming()
logging.shutdown()
