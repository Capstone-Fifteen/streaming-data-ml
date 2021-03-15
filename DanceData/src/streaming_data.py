import numpy as np
import pandas as pd
import os
import scipy
import math
import re
import json
import ast
import logging
import joblib
import sched, time

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

from tensorflow import keras

from datetime import datetime
from pathlib import Path

import src.utils as utils
import src.load as load
import src.signal_processing as sig_process
import src.segmentation as segment
import src.create_features as feature_ex
import src.clean_dataset as cleaner
import src.training_testing_models as model_exe


CUR_DIR    = utils.get_full_path_directory(mode="main")
DATA_DIR   = utils.get_full_path_directory(mode="data")
CONFIG_DIR = utils.get_full_path_directory(mode="configs")
LOGS_DIR   = utils.get_full_path_directory(mode="logs")
TEST_DIR   = utils.get_full_path_directory(mode="tests")

user_name = "Chi"
curr_exp  = "wk8"
key = f"{user_name}_{curr_exp}"

with open(f"{CONFIG_DIR}/default.json", "r") as f:
    params        = json.load(f)["meta_params"]
    sampling_freq = params["sampling_freq"]
    window_size   = params["window_size"]
    prefix        = params["prefix"]
    n_splits      = params["n_splits"]
print("Configuration for meta_variables:", params)

Activities_index_name = pd.read_table(f"{CONFIG_DIR}/activity_labels.txt", sep=" ", header=None).to_dict()[0]
Activities_name_index = dict((y,x) for x,y in Activities_index_name.items())
Member_index_name     = pd.read_table(f"{CONFIG_DIR}/user_labels.txt", sep=" ", header=None).to_dict()[0]
Member_name_index     = dict((y,x) for x,y in Member_index_name.items())
Experiment_ids        = ["wk6", "wk8"]

LOG_FILE_DIR = f"{LOGS_DIR}/{prefix}ws_{window_size}_split_{n_splits}_milestone2"
LOG_FILE_NAME = f"{LOG_FILE_DIR}.log"

EXP_DIR   = "_".join(Experiment_ids)
PROCESSED_DIR = utils.prepare_dir(f"{DATA_DIR}/ProcessedData/{EXP_DIR}")

logger = utils.setup_logger(LOG_FILE_DIR, LOG_FILE_NAME)
logger = getLogger(f"{LOG_FILE_NAME}")
logger.debug(f"Configuration for meta_variables: {params}")

models_name = dict()
# agreagating: > 300 columns
models_name[1]=["trained_model_dataset_1_model_bag_fold0.pkl",
                "trained_model_dataset_1_model_bayes_fold0.pkl",
                "trained_model_dataset_1_model_cart_fold0.pkl",
                "trained_model_dataset_1_model_et_fold0.pkl",
                "trained_model_dataset_1_model_gbm_fold0.pkl",
                "trained_model_dataset_1_model_knn_fold0.pkl",
                "trained_model_dataset_1_model_logReg_fold0.pkl",
                "trained_model_dataset_1_model_mlpSKlearn_fold0.pkl",
                "trained_model_dataset_1_model_rf_fold0.pkl",
                "trained_model_dataset_1_model_svm_fold0.pkl"]
# 15 columns
models_name[2]=["trained_model_dataset_3_fold0.h5"]
# 6 columnns
models_name[3]=["trained_model_dataset_5_fold0.h5"]

s = sched.scheduler(time.time, time.sleep)
row_time = (1/sampling_freq)*window_size/2.0
file_path = f"{DATA_DIR}/RawData/Chi-wk8(4).txt"

def model_list_predict(models_name, dataset_typ, X_train): 
    for model_name in models_name[dataset_typ]: 
        if model_name.endswith(".h5"): 
            model = keras.models.load_model(f'{TEST_DIR}/BestModel/{model_name}')
        else: 
            model = joblib.load(f'{TEST_DIR}/BestModel/{model_name}')
        
        pred = model.predict(X_train)
        max_pred = pred.argmax(axis=1)
        
        pred_df_string = pd.DataFrame(pred, columns=["Class "+str(i) for i in range(10)]).to_string().replace('\n', '\n\t')
        model_name_prd = f"{model_name} is used:\n"
        result = model_name_prd + pred_df_string + f"\nHighest probability class as {max_pred[0]}[{Activities_index_name[max_pred[0]]}]"
        
        logger.debug(result)
        print(result)
        
def predict_single_value(file_path, models_name): 
    time_sig_dic = sig_process.create_time_sig_dic(raw_dic_II)
    time_sig_dic_simple = sig_process.create_time_sig_dic(raw_dic_III,mode="simple")

    t_dic_win  = {f"t_W00000_{key}_act-1": time_sig_dic[key]}
    dic_win_simple = {f"W00000_{key}_act-1": time_sig_dic_simple[key]}

    f_dic_win  = {'f'+key[1:] : t_w1_df.pipe(sig_process.fast_fourier_transform) for key, t_w1_df in t_dic_win.items()}

    # dataset type 3
    Dataset_type_III = dic_win_simple

    # dataset type 2
    dic_win = feature_ex.create_window_df(t_dic_win, f_dic_win)
    Dataset_type_II  = dic_win 

    # dataset type 1
    Dataset_type_I   = feature_ex.Dataset_Generation_PipeLine(t_dic_win,f_dic_win).iloc[:, :-2]

    Dataset_type_II  = cleaner.scaling_DF(Dataset_type_II[list(Dataset_type_III.keys())[0]], 
                                          typ="streaming", key=list(Dataset_type_II.keys())[0])
    Dataset_type_III = cleaner.scaling_DF(Dataset_type_III[list(Dataset_type_III.keys())[0]], 
                                          typ="streaming", key=list(Dataset_type_III.keys())[0])

    X_train_I   = np.array(Dataset_type_I)

    X_train_II  = np.array(list(map(lambda x: x.to_numpy(), list(Dataset_type_II.values()))))
    X_train_II  = X_train_II.reshape(X_train_II.shape[0], X_train_II.shape[1], X_train_II.shape[2], 1)

    X_train_III = np.array(list(map(lambda x: x.to_numpy(), list(Dataset_type_III.values()))))
    X_train_III = X_train_III.reshape(X_train_III.shape[0], X_train_III.shape[1], X_train_III.shape[2], 1)

    # model_list_predict(models_name, 1, X_train_I)
    # model_list_predict(models_name, 2, X_train_II)
    model_list_predict(models_name, 3, X_train_III)
    
def streaming_main(sc): 
    try:         
        print("================================================================")
        contents = list(open(file_path, 'r', encoding="utf8"))
        df = load.import_raw_signals_v3(file_path, contents = contents[-window_size:])
    except: 
        print(f"File is not created yet! or file does not have more than {window_size} lines")
        pass
    s.enter(time, 0.5, streaming_main, (sc,))

s.enter(time, 0.5, streaming_main, (s,))
s.run()
