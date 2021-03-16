from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
from logging import getLogger
import csv
import io
import os
import json
from pathlib import Path
import math

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') # for better plots

import numpy as np
import pandas as pd
import seaborn as sns

# import shap
import tensorflow
from tensorflow import keras
import logging

import pickle

# shap.initjs()

# ====================================================== UTILITY functions for DIRECTORY computations ==========================================
def get_current_directory(mode='current'): 
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    return CUR_DIR

def get_full_path_directory(mode='main'): 
    path = Path(get_current_directory())
    path = Path(path.parent)
    if mode=="main": 
        return path
    elif mode=="data": 
        return os.path.join(path, "Training Data")
    elif mode=="configs": 
        return os.path.join(path, "configs")
    elif mode=="logs": 
        return os.path.join(path, "logs")
    elif mode=="tests": 
        return os.path.join(path, "Testing Data")
    
def prepare_dir(path): 
    os.makedirs(path, exist_ok=True)
    return path

def prepare_logs_dirs_training_testing(path): 
    prepare_dir(path)
    prepare_dir(f"{path}/images")
    prepare_dir(f"{path}/cross_validation")
    prepare_dir(f"{path}/history_pickle")
    prepare_dir(f"{path}/trained_model")
    prepare_dir(f"{path}/valid_oof")
    prepare_dir(f"{path}/test_oof")
    return path

# ====================================================== UTILITY functions for LOGGING ==========================================
class CsvFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_ALL)

    def format(self, record):
        self.writer.writerow([record.levelname, record.msg])
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()

    
def setup_logger(name, log_file, mode="w", level=logging.DEBUG, style="log_file"):
    """To setup as many loggers as you want
    Args: 
        name: name of logger
        log_file: location to store the log file
        level: logging level
    Returns: 
        a Logger object
    """
    head, tail = os.path.split(name)
    prepare_dir(head)
    
    if style=="log_file": 
        formatter = logging.Formatter("%(levelname)s: %(asctime)s: %(filename)s: %(funcName)s: \n%(message)s\n")
    elif style=="csv_file": 
        formatter = CsvFormatter()
    
    handler = logging.FileHandler(log_file, mode=mode)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.debug(f"Created {log_file}...")
    return logger

# ================================================ GET META VARIABLES ================================================
CUR_DIR    = get_full_path_directory(mode="main")
DATA_DIR   = get_full_path_directory(mode="data")
CONFIG_DIR = get_full_path_directory(mode="configs")
LOGS_DIR   = get_full_path_directory(mode="logs")
TEST_DIR   = get_full_path_directory(mode="tests")

prepare_dir(LOGS_DIR)
prepare_dir(CONFIG_DIR)
prepare_dir(DATA_DIR)
prepare_dir(TEST_DIR)

prepare_dir(f"{DATA_DIR}/RawData")
prepare_dir(f"{DATA_DIR}/ProcessedData")
prepare_dir(f"{DATA_DIR}/ProcessedData/RawData_csv")

with open(f"{CONFIG_DIR}/default.json", "r") as f:
    params        = json.load(f)["meta_params"]
    sampling_freq = params["sampling_freq"]
    window_size   = params["window_size"]
    prefix        = params["prefix"]
    n_splits      = params["n_splits"]
    
LOG_FILE_DIR = f"{LOGS_DIR}/{prefix}_freq_{sampling_freq}_ws_{window_size}_split_{n_splits}.log"
LOG_FILE_NAME = f"{LOG_FILE_DIR}.log"

logger = setup_logger(LOG_FILE_DIR, LOG_FILE_NAME)

logger = getLogger(LOG_FILE_NAME)
logger.debug(f"Configuration for meta_variables: {params}")


# ====================================================== UTILITY functions for PICKLE computations ==========================================
def pickle_save(path, obj): 
    with open(path, "wb") as f: 
        pickle.dump(obj, f)
        
def pickle_load(path): 
    with open(path, "rb") as f: 
        return pickle.load(f)

# ====================================================== UTILITY functions for VISUALIZATION  ================================================
def color_generator(i: int) -> str:
    l = ["#FFAF6D", "#DC4195", "#F1E898", "#6DCBB9", "#3E89C4", "#6F68CF"]
    return l[i]

def round_float(f, r = 0.000001):
    return float(Decimal(str(f)).quantize(Decimal(str(r)), rounding=ROUND_HALF_UP))

def round_list(l, r = 0.000001):
    return [round_float(f, r) for f in l]

def round_dict(d, r = 0.000001):
    return {key: round(d[key], r) for key in d.keys()}

def round_number(arg, r = 0.000001):
    if type(arg) == float or type(arg) == np.float64 or type(arg) == np.float32:
        return round_float(arg, r)
    elif type(arg) == list or type(arg) == np.ndarray:
        return round_list(arg, r)
    elif type(arg) == dict:
        return round_dict(arg, r)
    else:
        logger.error(f"Arg type {type(arg)} is not supported")
        return arg

# ====================================================== UTILITY functions for RAW DATA =============================================
def change_unit(df): 
    for i in df.columns: 
        if "Acc" in i: 
            df[i] = df[i]/(2**14)
        elif "Gyro" in i: 
            df[i] = df[i]/180.0*math.pi
    return df

def export_new_data(df, path): 
    df.to_csv(path_or_buf=path, na_rep='NaN',  
             columns=None, header=True, 
             index=False, mode='w', 
             encoding='utf-8',  
             line_terminator='\n')

def check_null(df):    
    nan_cols = df.columns[df.isnull().any()].tolist()
    logger.debug(f"Checking null: The NANs columns are {nan_cols}")
    df = df.drop(nan_cols, axis=1)
    return df

# ====================================================== UTILITY functions for FEATURES EXPLORATION ============================================
def look_up(Labels_Data_Frame, user_ID, exp_ID,activity_ID):
    """finding information based on user, experiment, activity in the labels dataframe.
    Args: 
        exp_ID  : string, usually indicates the name of the week 
        user_ID : string, the user name
        activity_ID: integer  the activity Identifier from 1 to 10 (10 included)
    Returns: 
        dataframe: A pandas Dataframe which is a part of Labels_Data_Frame contains 
        the activity ID ,the start point  and the end point  of this activity
    """
    return Labels_Data_Frame[(Labels_Data_Frame["experiment_number_ID"]==exp_ID)&
                             (Labels_Data_Frame["user_number_ID"]==user_ID)&
                             (Labels_Data_Frame["activity_number_ID"]==activity_ID)]

def remove_direction_feature(name): 
    subnames = name.split("_")
    if "Acc" in name: 
        return name.replace(subnames[-1], "Acc")
    elif "Gyro" in name: 
        return name.replace(subnames[-1], "Gyro")
    else: 
        return name
    
def assert_middle_feature(name, feature): 
    subnames = name.split("_")
    return "_".join(subnames[:-1]) + feature + "_" + subnames[-1]

# example: 679 ==> '00679'; 50 ==> '00050'
# it add '0's to the left of the input until the new lenght is equal to 5
def normalize5(number): 
    stre=str(number)
    if len(stre)<5:
        l=len(stre)
        for i in range(0,5-l):
            stre="0"+stre
    return stre 

# it add '0's to the left of the input until the new lenght is equal to 2
def normalize2(number):
    stre=str(number)
    if len(stre)<2:
        stre="0"+stre
    return stre

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

# ====================================================== UTLITY functions for training and testing ========================================
def check_class_balance(y_train, y_test, label2act, n_class = 6):
    c_train = Counter(y_train)
    c_test = Counter(y_test)

    for c, mode in zip([c_train, c_test], ["train", "test"]):
        logger.debug(f"{mode} labels")
        len_y = sum(c.values())
        for label_id in range(n_class):
            logger.debug(f"{label2act[label_id]} ({label_id}): {c[label_id]} samples ({c[label_id] / len_y * 100:.04} %)")

# ====================================================== GET GENERAL VARIABLES ======================================================
CUR_DIR    = get_full_path_directory(mode="main")
DATA_DIR   = get_full_path_directory(mode="data")
CONFIG_DIR = get_full_path_directory(mode="configs")
LOGS_DIR   = get_full_path_directory(mode="logs")
TEST_DIR   = get_full_path_directory(mode="tests")

with open(f"{CONFIG_DIR}/default.json", "r") as f:
    params        = json.load(f)["meta_params"]
    sampling_freq = params["sampling_freq"]
    window_size   = params["window_size"]
    prefix        = params["prefix"]
    n_splits      = params["n_splits"]

Activities_index_name = pd.read_table(f"{CONFIG_DIR}/activity_labels.txt", sep=" ", header=None).to_dict()[0]
Activities_name_index = dict((y,x) for x,y in Activities_index_name.items())
Member_index_name     = pd.read_table(f"{CONFIG_DIR}/user_labels.txt", sep=" ", header=None).to_dict()[0]
Member_name_index     = dict((y,x) for x,y in Member_index_name.items())
chosen_cols = ["AccX", "AccY", "AccZ", "GyroYaw", "GyroPitch", "GyroRoll"]

# ====================================================== VISUALIZATION functions for EDA ==================================================
def visualize_triaxial_signals(user_id,exp_id,act,sig_type,width,height,raw_dic,Labels_Data_Frame,LOG_DIR,pdf_write=None):
    """Visualize the acceleration and gyroscope signals
    Args:
         Data_frame: Data frame contains acc and gyro signals
         exp_id: integer from 1 to 61 (the experience identifier)
         width: integer the width of the figure
         height: integer the height of the figure
         sig_type: string  'acc' to visualize 3-axial acceleration signals or 'gyro' for 3-axial gyro signals
         act: possible values: string: 'all' (to visualize full signals), or integer from 1 to 10 to specify 
              the activity id to be visualized
    Returns:
        None
    """
    key = f"{user_id}_{exp_id}"
    if key not in raw_dic.keys(): 
        return
    data_frame = raw_dic[key]
    colors=['darkmagenta', 'salmon','cadetblue', 'darkmagenta', 'salmon','cadetblue']
    if act=='all':
        data_df=data_frame
        action_line = ""
    else: 
        action_line = f"{str(act)} ({Activities_index_name[act]})"
        indexes = []
        df_index = Labels_Data_Frame[(Labels_Data_Frame["experiment_number_ID"]==exp_id)&
                                     (Labels_Data_Frame["user_number_ID"]==Member_name_index[user_id])&
                                     (Labels_Data_Frame["activity_number_ID"]==act)][['Label_start_point','Label_end_point']]
        for i in df_index.values: 
            indexes.extend(range(i[0], i[1]+1))
        data_df=data_frame.iloc[indexes]
    if len(data_df)==0: 
        return
    if sig_type=="all": 
        data_array = []
        for col in chosen_cols: 
            data_array.append(data_df[col])
        fig, axs = plt.subplots(6,figsize=(width, height*6))
        
        if act=='all':
            title=f"all signals for all activities performed by user {user_id} in experience {exp_id}"
        elif act in range(0, 11):
            title=f"all signals of experience {exp_id} while user {user_id} was performing activity: {action_line}"
        axs[0].set_title(title)
        for i in range(len(chosen_cols)): 
            axs[i].plot(data_df["time"].values, data_array[i], linewidth=2, color=colors[i], marker='o', markersize=2)
            axs[i].set_ylabel(chosen_cols[i])
        axs[5].set_xlabel('Time in seconds[s]')
    else: 
        if sig_type=='acc':# if the columns to be visualized are acceleration columns
            indexes = [0, 1, 2]
            figure_Ylabel='Acceleration in 1g'
            if act=='all':
                title=f"acceleration signals for all activities performed by user {user_id} in experience {exp_id}"
            elif act in range(0, 11):
                title=f"acceleration signals of experience {exp_id} while user {user_id} was performing activity: {action_line}"
        elif sig_type=='gyro':# if the columns to be visualized are gyro columns
            indexes = [3, 4, 5]
            figure_Ylabel='Angular Velocity[rad/s]'
            if act=='all':
                title=f"gyroscope signals for all activities performed by user {user_id} in experience {exp_id}"
            elif act in range(0, 11):
                title=f"gyroscope signals of experience {exp_id} while user {user_id} was performing activity: {action_line}"
        time=data_df["time"].values
        fig = plt.figure(figsize=(width,height))
        for i in indexes: 
            _ = plt.plot(time,data_df[chosen_cols[i]],color=colors[i],label=chosen_cols[i], linewidth=2)
        _ = plt.ylabel(figure_Ylabel) # set Y axis info 
        _ = plt.xlabel('Time in seconds[s]') # Set X axis info (same label in all cases)
        _ = plt.title(title) # Set the title of the figure
        _ = plt.legend(loc="upper left")# upper left corner
    if pdf_write: 
        pdf_write.savefig(fig, bbox_inches='tight', pad_inches=0.3)
    else: 
        fig.savefig(os.path.join(LOG_DIR, f"user_{user_id}_exp_id_{exp_id}_act{act}_sig_{sig_type}.pdf"), bbox_inches='tight', pad_inches=0.3)
    plt.close()

# ====================================================== VISUALIZATION functions for training and testing =======================================
# def plot_feature_importance(models,num_features,cols,importance_type="gain",path="importance.pdf",figsize=(18, 10),max_display=-1, pdf_write=None):
#     """
#     Args:
#         importance_type: chosen from "gain" or "split"
#     """
#     importances = np.zeros((len(models), num_features))
#     for i, model in enumerate(models):
#         importances[i] = model.feature_importance(importance_type=importance_type)
#     importance = np.mean(importances, axis=0)
#     importance_df = pd.DataFrame({"Feature": cols, "Value": importance})
#     importance_df = importance_df.sort_values(by="Value", ascending=False)[:max_display]
#     fig = plt.figure(figsize=figsize)
#     sns.barplot(x="Value", y="Feature", data=importance_df)
#     plt.title("Feature Importance (avg over folds)")
#     plt.tight_layout()
#     if pdf_write: 
#         pdf_write.savefig(fig, bbox_inches='tight', pad_inches=0.3)
#     else: 
#         fig.savefig(path, bbox_inches='tight', pad_inches=0.3)
#     plt.close()

# def plot_shap_summary(models, X_train, class_names, path="shap_summary_plot.pdf",max_display=None, pdf_write=None):
#     shap_values_list = []
#     for model in models:
#         explainer = shap.TreeExplainer(
#             model,
#             num_iteration=model.best_iteration,
#             feature_perturbation="tree_path_dependent",
#         )
#         shap_value_oof = explainer.shap_values(X_train)
#         shap_values_list.append(shap_value_oof)
#     shap_values = [np.zeros(shap_values_list[0][0].shape) for _ in range(len(class_names))]
#     for shap_value_oof in shap_values_list:
#         for i in range(len(class_names)):
#             shap_values[i] += shap_value_oof[i]
#     for i in range(len(class_names)):
#         shap_values[i] /= len(models)
#     shap.summary_plot(
#         shap_values,
#         X_train,
#         max_display=max_display,
#         class_names=class_names,
#         color=color_generator,
#         show=False,
#     )
#     if pdf_write: 
#         pdf_write.savefig(fig, bbox_inches='tight', pad_inches=0.3)
#     else: 
#         plt.savefig(path, bbox_inches='tight', pad_inches=0.3)
#     plt.close()

def plot_confusion_matrix(cms,labels=None,path="confusion_matrix.pdf",pdf_write=None,clf_name="",option="general_method"):
    """Plot confusion matrix"""
    if option=="general_method": 
        columns = ["train", "valid", "test"]
    elif option=="no_test": 
        columns = ["train", "valid"]
    elif option=="no_cv": 
        columns = ["train", "test"]
        
    cms = [np.round(np.mean(cms[mode], axis=0), 2) for mode in columns]
    fig, ax = plt.subplots(3, figsize=(10, 24))
    for i, (confusion_matrix, mode) in enumerate(zip(cms, columns)):
        g = sns.heatmap(
            confusion_matrix,
            annot=True,
            cmap="RdPu",
            square=True,
            vmin=0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax[i],
        )
        ax[i].set_xlabel("Predicted label")
        ax[i].set_ylabel("True label")
        ax[i].set_title(f"Normalized confusion matrix for {clf_name.upper()} - {mode}")
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation = 45) 
    plt.tight_layout()
    if pdf_write: 
        pdf_write.savefig(fig, bbox_inches='tight', pad_inches=0.3)
    else: 
        fig.savefig(path, bbox_inches='tight', pad_inches=0.3)
    plt.close()

def plot_model(model, path):
    if not os.path.isfile(path):
        keras.utils.plot_model(model, to_file=path, show_shapes=True)

def plot_learning_history(fit, metric="accuracy", path="history.pdf", pdf_write=None):
    """Plot learning curve
    Args:
        fit (Any): History object
        path (str, default="history.pdf")
    """
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(18, 5))
    axL.plot(fit.history["loss"], label="train")
    axL.plot(fit.history["val_loss"], label="validation")
    axL.set_title("Loss")
    axL.set_xlabel("epoch")
    axL.set_ylabel("loss")
    axL.legend(loc="upper right")
    axR.plot(fit.history[metric], label="train")
    axR.plot(fit.history[f"val_{metric}"], label="validation")
    axR.set_title(metric.capitalize())
    axR.set_xlabel("epoch")
    axR.set_ylabel(metric)
    axR.legend(loc="upper right")
    if pdf_write: 
        pdf_write.savefig(fig, bbox_inches='tight', pad_inches=0.3)
    else: 
        fig.savefig(path, bbox_inches='tight', pad_inches=0.3)
    plt.close()
