import json
import re
import os
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import joblib

plt.style.use('fivethirtyeight')  # for better plots

# ================================================ GET META VARIABLES ==================================================

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

LOG_DIR = utils.prepare_dir(f"{LOGS_DIR}/{prefix}_freq_{sampling_freq}_ws_{window_size}_clean_dataset")

logger = getLogger(f"{LOGS_DIR}/{prefix}_freq_{sampling_freq}_ws_{window_size}_split_{n_splits}.log")
separator = "_______________________________"


# ====================================================== UTILITY functions =============================================
def num_row_user_act(Df):
    """returns a table includes the number of windows per each tuple(user_id , activity id) included in the dataset 
    Args: 
        Df: dataframe
    Returns: 
        A new dataframe
    """
    user_Ids = sorted(Df['user_Id'].unique())  # extracting and sorting unique user ids
    activity_Ids = sorted(Df['activity_Id'].unique())  # extracting and sorting unique activity ids
    act_columns = ['Activity ' + str(int(Id)) for Id in activity_Ids]  # defining column names used in output table
    users_index = ['User ' + str(Id) for Id in user_Ids]  # defining rows names used in output table
    # counting the number of windows per each tuple(user_id,activity_id)
    # store these values in 2D numpy array
    data = np.array([[len(Df[(Df["user_Id"] == user_ID) & (Df["activity_Id"] == activity_ID)])
                      for activity_ID in activity_Ids] for user_ID in user_Ids])
    # Create a pandas dataframe from the array above
    win_per_act_per_user = pd.DataFrame(data=data, columns=act_columns, index=users_index)
    return win_per_act_per_user  # returns the dataframe


def visualize_column(Df, column, Dataset_name, LOG_DIR, Activities_index_name, mode="original", pdf_write=None):
    """
    Args: 
        Df: dataframe
        column: activity ids
        Dataset_name: dataset numbering
        LOG_DIR : logging directory
        Activities_index_name: dictionary of activities name
        mode: dataset mode
        pdf_write : PdfPages object
    Returns:
        the weights activity and visualize the distribution of a column
    """
    labels = sorted(Df[column].unique())  # extracting and sorting activity unique ids
    height = int((len(labels) + 2) / 2)
    Als_dict = {key: len(Df[Df[column] == key]) for key in labels}  # counting the number of windows per activity
    data = [Als_dict[key] for key in labels]  # sorting these numbers
    weights = np.array(data) / float(np.array(data).sum())  # calculating weights of each activity
    columns = [f"Activity {str(int(key))} ({Activities_index_name[int(key)]})" for key in
               labels]  # defining columns of weights' table

    Df_weights = pd.DataFrame(data=None, columns=columns)  # defining an empty dataframe with column names
    Df_weights.loc['Percentage'] = weights  # appending weights row

    logger.debug("_____ The Percentage of each activity _____")
    logger.debug(Df_weights.to_string().replace('\n', '\n\t'))

    fig, ax = plt.subplots(figsize=(14, height))
    plt.barh(columns, data)  # ploting activity distribution
    plt.xlabel('Activity Labels')  # set X axis info
    plt.ylabel('Number of Data points')  # set Y axis info
    plt.title(f'Number of Data points per activity for {Dataset_name}')  # set the figure's title

    if pdf_write:
        pdf_write.savefig(fig, bbox_inches='tight', pad_inches=0.3)
    else:
        fig.savefig(os.path.join(LOG_DIR, f"EDA_{Dataset_name}[{mode}].pdf"), bbox_inches='tight', pad_inches=0.3)
    plt.close()

def filter_mode(name, mode_list): 
    words_re = re.compile("|".join(mode_list))
    return words_re.search(name)
    

# ====================================================== DATA EXPLORATION ==============================================
def data_exploration_pipeline(Dataset, typ, outliers, LOG_DIR, Activities_index_name, mode="original", visualize=True, pdf_write=None):
    """Data exploration pipeline
    Args: 
        Dataset: a pandas dataframe can be a full dataset (I or II), cleaned dataset(I or II or III),
        outliers dataset (I or II)
        typ: integer type of the dataset possible values
        outliers: Boolean if true dataset we are dealing with is an outlier dataset(contain outlier values)
        LOG_DIR : logging directory
        Activities_index_name: dictionary of activities name
        mode: dataset mode
        visualize: boolean indicating whether to create visualization or not
        pdf_write : PdfPages object
    Returns:
        None
    """
    if not outliers:  # in case we are not dealing with outliers datasets
        # Adapting the dataset name switch the typ
        if typ == 1:
            Dataset_name = "Dataset type I "
        if typ == 2:
            Dataset_name = "Dataset type II "
        if typ == 3:
            Dataset_name = "Dataset type III "
        if typ == 4:
            Dataset_name = "Dataset type IV "
    else:  # in case we are dealing with outliers
        # adapting the dataset names switch the case
        if typ == 1:
            Dataset_name = "Outliers of Dataset type I "
        if typ == 2:
            Dataset_name = "Outliers of Dataset type II "

    logger.debug(f"{separator} Starting to explore {Dataset_name} {separator}")
    # general info about the dataset: number of rows and columns
    logger.debug(
            Dataset_name + 'has a shape of: ' + str(Dataset.shape[0]) + ' rows and ' + str(
                    Dataset.shape[1]) + ' columns')

    Stats = num_row_user_act(Dataset)  # generate number of windows per each tuple (user,activity)
    logger.debug("Number of windows per user and per each activity:")
    logger.debug(Stats.to_string().replace('\n', '\n\t'))

    logger.debug("Statistics of table above:")
    logger.debug(Stats.describe().to_string().replace('\n', '\n\t'))
    
    if visualize: 
        visualize_column(Dataset, "activity_Id", Dataset_name, LOG_DIR, 
                         Activities_index_name, mode=mode, pdf_write=pdf_write)


# ====================================================== HANDLING OUTLIERS =============================================
def extract_drop_outliers(Df, threshold, typ, LOG_DIR, Activities_index_name, visualize=True, pdf_write=None):
    """Drop outliers for Dataset I and II 
    Args: 
        Df: pandas dataframe (Dataset type I or Dataset type II)
        threshold: integer : if the number of features detected as outliers in row exceeds
                             the threshold the row will be considered as "outlier row"
        typ: Dataset type
        LOG_DIR : logging directory
        pdf_write : PdfPages object
    Returns:
        Clean dataset without outliers
    """
    max_range = len(Df["activity_Id"].unique())  # number of unique activities in Df
    columns = Df.columns  # column names of the dataset
    outliers = {}  # dictionary will contain number of outliers per row . keys are rows' indexes

    for i in range(1, max_range + 1):  # iterate throw each activity type in the dataset
        Df_A = Df[Df['activity_Id'] == i]  # select rows related to this activity

        for column in columns[:-2]:  # iterate throw features columns only in Df_A
            q1 = Df_A[column].describe()['25%']  # the value of the first quartile of a column in Df_A
            q3 = Df_A[column].describe()['75%']  # the value of the third quartile of a column in Df_A
            low_threshold = q1 - 1.5 * (q3 - q1)  # define low threshold to detect bottom outliers of a column
            high_threshold = q3 + 1.5 * (q3 - q1)  # define high threshold to detect top outliers of a column

            for e in Df_A.index:  # iterate throw Df_A indexes
                if Df[column].iloc[e] > high_threshold or Df[column].iloc[e] < low_threshold:  # if value is outlier
                    if e in outliers.keys():  # if the row index is already exist in outliers dictionary
                        outliers[e] = outliers[e] + 1  # increase the number of outliers for this row
                    else:  # if the row index does not exist yet in  outliers dic keys
                        outliers[e] = 1  # add the key with outlier number =1
    indexes = np.array(sorted(outliers.keys()))  # rows indexes contain outlier values sorted from low to high
    values = np.array([outliers[indexes[i]] for i in range(len(indexes))])  # number of outliers related to each row
    indexes_drooped = indexes[
        values > threshold]  # store indexes having number of outliers exceeding the threshold in a list
    if len(indexes_drooped) > 0:
        # Build outliers dataframe using row's indexes
        outliers_data = np.array([list(Df.iloc[indexes_drooped[i]]) for i in range(len(indexes_drooped))])
        outliers_Df = pd.DataFrame(data=outliers_data, columns=columns)
        
    # generate the clean dataframe by dropping outliers from the original dataframe
    clean_Df = Df.drop(indexes_drooped, 0, )
    
    # adapting the name of the dataset switch the case
    if typ == 1:
        dataset_name = 'Dataset type I'
    if typ == 2:
        dataset_name = "Dataset type II"
    
    # report
    logger.debug(
            f"{separator} Original Data Frame info...{separator}\nNumber of rows in the original dataframe "
            f"{dataset_name}: {len(Df)}")  # original dataset length

    visualize_column(Df, 'activity_Id', dataset_name, LOG_DIR, Activities_index_name, mode="original",
                     pdf_write=pdf_write)  # activity distribution of the original dataset

    logger.debug(
            f"{separator} Outliers info...{separator}\nA row is considered as outlier if the number of its outliers "
            f"exceeds: {str(threshold)}\nNumber of rows dropped : {len(indexes_drooped)}') # number of rows considered "
            f"as outliers")

    if len(indexes_drooped) > 0:
        data_exploration_pipeline(outliers_Df, typ, True, LOG_DIR, 
                                  Activities_index_name, mode="outliers", visualize=visualize, pdf_write=pdf_write)

    logger.debug(
            f"{separator} Cleaned+{dataset_name} Dataframe info...{separator}\nNumber of rows in the clean dataframe " + \
            f"{dataset_name}: {len(clean_Df)}")  # clean dataframe info

    data_exploration_pipeline(clean_Df, typ, False, LOG_DIR, 
                              Activities_index_name, mode="cleaned", visualize=visualize, pdf_write=pdf_write)

    return clean_Df  # return the clean dataset

# ====================================================== SCALING AND TRANSFORMATION ====================================
def scaling_array(oneD_signal, scaler="normalize"):
    sc=None
    if scaler == "normalize":
        sc = StandardScaler()
        signal_standardScaler = sc.fit_transform(signal)
        
    elif scaler == "min-max":
        # inputs: 1D numpy array (one column)
        maximum = oneD_signal.max()  # maximum of the column
        minimum = oneD_signal.min()  # min value of the column
        Difference = float(maximum - minimum)  # max-min
        # scaling formula: 2 * (x_i-minimum)/(maximum-minimum)
        # apply the scaling formula to each value in the column
        scaled_signal = np.array(
                [((2 * float(oneD_signal[i]) - minimum) / float(Difference)) for i in range(len(oneD_signal))])
    
    # return the scaled array
    return scaled_signal


def scaling_DF(data_frame, user_key="", typ=None, key=""):
    # input : pandas dataframe (clean datasets type I or II)
    columns = data_frame.columns  # column names
    if typ == "window":
        scaled_array = np.apply_along_axis(scaling_array, 0, np.array(data_frame[columns]))
        return pd.DataFrame(data=scaled_array, columns=columns)
        
    elif typ == "streaming": 
        dict_res = dict()
        scaled_array = np.apply_along_axis(scaling_array, 0, np.array(data_frame[columns]))
        dict_res[key] = pd.DataFrame(data=scaled_array, columns=columns)
        return dict_res        
    
    else:
        # apply the scaling function to each feature columns only
        scaled_array = np.apply_along_axis(scaling_array, 0, np.array(data_frame[columns[:-2]]))
        # build the scaled dataset
        scaled_df = pd.DataFrame(data=scaled_array, columns=columns[:-2])
        # the user and activity ids columns
        scaled_df['activity_Id'] = np.array(data_frame['activity_Id'])
        scaled_df['user_Id'] = np.array(data_frame['user_Id'])
        return scaled_df  # return the scaled dataset


# ====================================================== SPLITTING DATA for aggregated data ============================
def create_training_testing_data(scaled_Df, train_users, test_users, typ, LOG_DIR, 
                                 Activities_index_name, visualize=True, pdf_write=None, mode="dual"):
    """Creating training and testing data based on train and test users
    Args: 
        scaled_Df : pandas dataframe already scaled
        train_users: list of integers contains train user ids 
        test_users: list of integers contains test user ids
        LOG_DIR : logging directory
        pdf_write : PdfPages object
        typ        : integer from 1 to 3 (depending on the dataset type)
    Returns: 
        X_train, X_test, y_train, y_test

    """
    if mode=="single": 
        array_train = np.array([np.array(scaled_Df.iloc[i])for i in range(len(scaled_Df))])
        columns = scaled_Df.columns
        
        Df_train = pd.DataFrame(data=array_train, columns=columns)
        Df_train_features = Df_train[columns[:-2]]
        Df_train_labels = Df_train[columns[-2:-1]]
        
        X_train = np.array(Df_train_features)
        
        y_train = np.array(Df_train_labels['activity_Id'])
        y_train = y_train.astype('int')
        return [X_train, None, y_train, None]
    
    elif mode=="dual": 
        array_train = np.array([np.array(scaled_Df.iloc[i])
                                for i in range(len(scaled_Df)) if scaled_Df['user_Id'].iloc[i] in train_users])
        array_test = np.array([np.array(scaled_Df.iloc[i])
                               for i in range(len(scaled_Df)) if scaled_Df['user_Id'].iloc[i] in test_users])
        columns = scaled_Df.columns
        Df_train = pd.DataFrame(data=array_train, columns=columns)
        Df_test = pd.DataFrame(data=array_test, columns=columns)
        Df_train_features = Df_train[columns[:-2]]
        Df_train_labels = Df_train[columns[-2:-1]]
        Df_test_features = Df_test[columns[:-2]]
        Df_test_labels = Df_test[columns[-2:-1]]
        X_train = np.array(Df_train_features)
        X_test = np.array(Df_test_features)
        y_train = np.array(Df_train_labels['activity_Id'])
        y_test = np.array(Df_test_labels['activity_Id'])
        y_train = y_train.astype('int')
        y_test = y_test.astype('int')
        if typ == 1:
            Dataset_name = "Dataset type I "
        if typ == 2:
            Dataset_name = "Dataset type II "
        if typ == 3:
            Dataset_name = "Dataset type III "
        if typ == 4:
            Dataset_name = "Dataset type IV "
        if typ == 3:
            Dataset_name = "Dataset type III"
        if typ == 4:
            Dataset_name = "Dataset type IV"
        if typ == 5:
            Dataset_name = "Dataset type V"
        if typ == 6:
            Dataset_name = "Dataset type VI"
        if visualize: 
            visualize_column(Df_train, 'activity_Id', Dataset_name, LOG_DIR, Activities_index_name, mode="agg_training", 
                             pdf_write=pdf_write)
            visualize_column(Df_test, 'activity_Id', Dataset_name, LOG_DIR, Activities_index_name, mode="agg_testing", 
                             pdf_write=pdf_write)

        return [X_train, X_test, y_train, y_test]


# ====================================================== SPLITTING DATA for window data ================================
def create_window_labels_info_df(df_meta_dict):
    keys = list(df_meta_dict.keys())
    df = pd.DataFrame(keys, columns=["key"])
    df['activity_id'] = df['key'].apply(lambda i: int(extract_info_window_key(i, "activity")))
    return df


# W00000_Jeff_wk4_act02
def extract_info_window_key(key, mode="user"):
    """
    Extract window key information
    :param key: window key
    :type key: string
    :param mode: type inside window we want to extract
    :type mode: string
    :return:
    :rtype: string or int
    """
    names = key.split("_")
    if mode == "window":
        return names[0]
    elif mode == "user":
        return names[1]
    elif mode == "experiment":
        return names[2]
    elif mode == "activity":
        return int(names[3][-2:])
    elif mode == "key": 
        return "_".join([names[1], names[2]])


def create_window_training_testing_data(df_meta_dict, train_key_users, test_key_users, typ, LOG_DIR, 
                                        Activities_index_name, visualize=True, pdf_write=None, mode="dual"):
    """Creating window-based training and testing data based on train and test users
    Args: 
        df_meta_dict : dictionary of pandas dataframes already scaled
        train_users: list of integers contains train user ids 
        test_users: list of integers contains test user ids
        typ        : integer from 1 to 3 (depending on the dataset type)
        LOG_DIR : logging directory
        pdf_write : PdfPages object
    Returns:
        X_train, X_test, y_train, y_test
    """
    if mode=="single": 
        train_dict = df_meta_dict
        
        X_win_train = np.array(list(map(lambda x: x.to_numpy(), list(train_dict.values()))))
        y_win_train = np.array([int(extract_info_window_key(i, "activity")) for i in train_dict.keys()])
        
        print(X_win_train.shape)
        X_win_train = X_win_train.reshape(X_win_train.shape[0], X_win_train.shape[1], X_win_train.shape[2], 1)
        
        return [X_win_train, None, y_win_train, None]

    elif mode=="dual": 
        train_dict = dict(
            filter(
                lambda elem: any(item in extract_info_window_key(elem[0], "key") for item in train_key_users), 
                    df_meta_dict.items()))
        test_dict = dict(
            filter(
                lambda elem: any(item in extract_info_window_key(elem[0], "key") for item in test_key_users), 
                df_meta_dict.items()))

        X_win_train = np.array(list(map(lambda x: x.to_numpy(), list(train_dict.values()))))
        X_win_test = np.array(list(map(lambda x: x.to_numpy(), list(test_dict.values()))))

        y_win_train = np.array([int(extract_info_window_key(i, "activity")) for i in train_dict.keys()])
        y_win_test = np.array([int(extract_info_window_key(i, "activity")) for i in test_dict.keys()])

        X_win_train = X_win_train.reshape(X_win_train.shape[0], X_win_train.shape[1], X_win_train.shape[2], 1)
        X_win_test = X_win_test.reshape(X_win_test.shape[0], X_win_test.shape[1], X_win_test.shape[2], 1)

        if typ == 1:
            Dataset_name = "Dataset type I "
        if typ == 2:
            Dataset_name = "Dataset type II "
        if typ == 3:
            Dataset_name = "Dataset type III "
        if typ == 4:
            Dataset_name = "Dataset type IV "
        if typ == 3:
            Dataset_name = "Dataset type III"
        if typ == 4:
            Dataset_name = "Dataset type IV"
        if typ == 5:
            Dataset_name = "Dataset type V"
        if typ == 6:
            Dataset_name = "Dataset type VI"

        if visualize: 
            visualize_column(create_window_labels_info_df(train_dict), 'activity_id', Dataset_name, LOG_DIR, Activities_index_name, 
                             mode="window_training", pdf_write=pdf_write)
            visualize_column(create_window_labels_info_df(test_dict), 'activity_id', Dataset_name, LOG_DIR, Activities_index_name,
                             mode="window_testing", pdf_write=pdf_write)

        return [X_win_train, X_win_test, y_win_train, y_win_test]
