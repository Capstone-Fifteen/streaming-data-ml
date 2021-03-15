# ================================================ GET META VARIABLES ==================================================
import json
import math
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display
from matplotlib.backends.backend_pdf import PdfPages
from typing import Any, Dict, List, Optional
from logging import getLogger
from logging import basicConfig, getLogger, StreamHandler, DEBUG, WARNING

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

# ================================================ TRADITIONAL MODELS ==================================================
import tensorflow
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ================================================ MODEL EVALUATIONS ===================================================
from sklearn.metrics import (accuracy_score, confusion_matrix as cm, f1_score, precision_score, recall_score)
from sklearn.model_selection import (StratifiedKFold)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # import Multi-layer perceptron classifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ================================================ NEURAL NETWORK MODELS ===============================================
tensorflow.random.set_seed(0)

from tensorflow import keras
from tensorflow.keras import backend as K, optimizers
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, LSTM, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.callbacks import CSVLogger

# ================================================ KERAS CALLBACKS ===================================================
class F1Callback(Callback):
    """Plot f1 value of every epoch"""
    def __init__(self,model: Model,path_f1_history: str,
                 X_tr: np.ndarray,y_tr: np.ndarray,X_val: np.ndarray,y_val: np.ndarray):
        self.model = model
        self.path_f1_history = path_f1_history
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_val = X_val
        self.y_val = y_val
        self.tr_fscores: List[float] = []  # train f1 of every epoch
        self.val_fscores: List[float] = []  # valid f1 of every epoch
    
    
    def on_epoch_end(self, epoch, logs):
        tr_pred = self.model.predict(self.X_tr)
        tr_macro_f1 = f1_score(self.y_tr.argmax(axis=1), tr_pred.argmax(axis=1), average="macro")
        self.tr_fscores.append(tr_macro_f1)
        val_pred = self.model.predict(self.X_val)
        val_macro_f1 = f1_score(self.y_val.argmax(axis=1), val_pred.argmax(axis=1), average="macro")
        self.val_fscores.append(val_macro_f1)
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.plot(self.tr_fscores, label="f1_score for training")
        ax.plot(self.val_fscores, label="f1_score for validation")
        ax.set_title("model f1_score")
        ax.set_xlabel("epoch")
        ax.set_ylabel("f1_score")
        ax.legend(loc="upper right")
        fig.savefig(self.path_f1_history)
        plt.close()

        
class PeriodicLogger(Callback):
    """Logging history every n epochs"""
    def __init__(self, metric="accuracy", logger=None, verbose=5, epochs=None):
        self.metric = metric
        self.verbose = verbose
        self.epochs = epochs
        self.logger = logger
    
    
    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if self.verbose == 0: 
            return
        if epoch % self.verbose == 0:
            msg = " - ".join([
                f"Epoch {epoch}/{self.epochs}",
                f"loss: {utils.round_number(logs['loss'], 0.00001)}",
                f"{self.metric}: {utils.round_number(logs[self.metric], 0.00001)}",
                f"val_loss: {utils.round_number(logs['val_loss'], 0.00001)}",
                f"val_{self.metric}: {utils.round_number(logs[f'val_{self.metric}'], 0.00001)}",])
            self.logger.debug(msg)
            print(msg)

        
def create_callback(
    model: Model, path_chpt, logger=None, patience=200, metric="accuracy", verbose=5, epochs=None):
    """callback settinngs
    Args:
        model (Model)
        path_chpt (str): path to save checkpoint
    Returns:
        callbacks (List[Any]): List of Callback
    """
    callbacks = []
    
    callbacks.append(
        EarlyStopping(monitor="val_loss", min_delta=0, patience=patience, verbose=1, mode="min", restore_best_weights=True))
    
    callbacks.append(
        ModelCheckpoint(filepath=f"{path_chpt}.h5", save_best_only=True))
    
    callbacks.append(
        PeriodicLogger(metric=metric, logger=logger, verbose=verbose, epochs=epochs))
    
    callbacks.append(
        CSVLogger(f"{path_chpt}.csv", append=True, separator=','))
    
    return callbacks

# ================================================ SCORING functions ===================================================

dataset_numbering = {1: "I", 2: "II", 
                     3: "III", 3.1: "III.1", 3.2: "III.2", 3.3: "III.3", 3.4: "III.4", 3.5: "III.5",
                     4: "IV", 5: "V", 
                     6: "VI", 6.1: "VI.1", 6.2: "VI.2", 6.3: "VI.3", 6.4: "VI.4", 6.5: "VI.5"}

criteria = ["accuracy", "precision", "recall", "f1_score", "log_loss", "cm", "cm_norm", "per_class_f1"]
modes = ["train", "valid", "test", "macro averaged"]
separator = "=========================================================="


def initialize_scoring():
    scores = dict()
    for typ in dataset_numbering.keys():
        scores[typ] = dict()
        for mode in modes:
            scores[typ][mode] = dict()
            for cri in criteria:
                scores[typ][mode][cri] = []
        for cri in ["cm_norm"]:
            scores[typ][cri] = dict()
            for mode in modes:
                scores[typ][cri][mode] = []
    return scores

# ================================================ UTILITY functions ===================================================
def get_training_testing_data(train_test_files_dic, typ=1):
    files = train_test_files_dic[typ]
    return files[0], files[1], files[2], files[3]


def is_NN_model(clf_name):
    if clf_name in ["cnn", "mlp", "deep_conv_lstm"]:
        return True
    else:
        return False


def make_log_dir(INFO, clf_name="traditional_models", prefix="", option="general_method"):
    utils.prepare_dir(f"{LOGS_DIR}/{INFO}")
    utils.prepare_dir(f"{LOGS_DIR}/{INFO}/AllModels")
    
    if is_NN_model(clf_name):
        return f"{LOGS_DIR}/{INFO}/{prefix}_freq_{sampling_freq}_ws_{window_size}_split_{n_splits}_{clf_name}[{option}]"
    else:
        return f"{LOGS_DIR}/{INFO}/{prefix}_freq_{sampling_freq}_ws_{window_size}_split_{n_splits}_traditional_models[{option}]"


def get_dataset_type(clf_name, intended_dataset_number, mode="test", option=""):
    result_list = []
    if mode=="no_test": 
        if is_NN_model(clf_name):
            result_list = [i for i in intended_dataset_number if math.floor(i) in [5, 6]]
        else:
            result_list = [i for i in intended_dataset_number if math.floor(i) in [4]]
    else: 
        if is_NN_model(clf_name):
            result_list = [i for i in intended_dataset_number if math.floor(i) in [2, 3]]
        else:
            result_list = [i for i in intended_dataset_number if math.floor(i) in [1]]
    if option=="max": 
        return max(result_list)
    else: 
        return result_list

def check_null(df):
    nan_cols = df.columns[df.isnull().any()].tolist()
    df = df.drop(nan_cols, axis=1)
    return df


def get_freq_along_axis(array, axis=1):
    most_freq_elem = []
    if axis == 0:
        for row in array:
            elements, counts = np.unique(row, return_counts=True)
            most_freq_elem.append(elements[np.argmax(counts)])
    elif axis == 1:
        for col in array.T:
            elements, counts = np.unique(col, return_counts=True)
            most_freq_elem.append(elements[np.argmax(counts)])
    return np.array(most_freq_elem)


def export_summary(df_1, df_2, prefix, model_names, INFO, export_mode="test"):
    model_name_agg = "_".join(model_names)
    writer_summary = pd.ExcelWriter(
            f"{LOGS_DIR}/{INFO}/AllModels/{prefix}_freq_{sampling_freq}_ws_{window_size}_split_{n_splits}_{model_name_agg}[{export_mode}].xlsx",
            engine='xlsxwriter')
    try: 
        df_1 = df_1.sort_values(by=["Dataset", "Mode", "Classifier"], ignore_index=True)
        df_2 = df_2.sort_values(by=["Dataset", "Classifier", "Fold"], ignore_index=True)
        df_2.columns = ['_'.join(col) for col in df_2.columns]

        df_1.to_excel(writer_summary, sheet_name=f'summary', index=False)
        df_2.to_excel(writer_summary, sheet_name=f'summary_per_fold', index=False)

        df_1 = df_1.set_index(["Dataset", "Mode", "Classifier"])
        df_2 = df_2.set_index(["Dataset_", "Classifier_", "Fold_"])

        df_1.to_excel(writer_summary, sheet_name=f'summary_MultiIndex')
        df_2.to_excel(writer_summary, sheet_name=f'summary_per_fold_MultiIndex')

        writer_summary.save()
        utils.pickle_save(
            f"{LOGS_DIR}/{INFO}/AllModels/{prefix}_freq_{sampling_freq}_ws_{window_size}_"
            f"split_{n_splits}_classifiers_summary_meta[{export_mode}].pickle", df_1)
        utils.pickle_save(
            f"{LOGS_DIR}/{INFO}/AllModels/{prefix}_freq_{sampling_freq}_ws_{window_size}_"
            f"split_{n_splits}_cv_df_meta[{export_mode}].pickle", df_2)
    except: 
        display(df_1)
        display(df_2)


# ================================================ CONFUSION MATRIX ====================================================
# Define the adpted confusion matrix
def full_confusion_matrix(Df):
    """Customized Confusion matrix
    Args: 
        Df : pandas dataframe, the contingency table resulted from the confusion matrix defined earlier as cm
    Returns: 
        Confusion 
    """
    columns = Df.columns  # activity names
    additional_cols = ['data points number', 'precision %', 'sensitivity %', 'specificity %', 'f1_score']
    new_columns = list(columns)+additional_cols
    
    # create the index from the same old columns add an other row called total
    new_index = list(columns)+['Total']
    
    # intialize the confustion matrix dataframe
    new_Df = pd.DataFrame(data=None, columns=new_columns, index=new_index)
    
    # intialize values
    total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
    total_data_points_number = 0  # total number of datapoints
    
    for column in columns:
        TP = Df.loc[column][column]  # extract true positives from the contingency table
        FN = Df.loc[column].sum()-TP  # calculate FN(false negatives)
        FP = Df[column].sum()-TP  # calculate FP(false positives)
        TN = (Df.sum()).sum()-TP-FN-FP  # calculate TN(true negatives)
        class_data_points = TP+FN  # number of datapoints per activity
        
        # precision score in %
        precision = TP/float(TP+FP)*100
        
        # Recall or sensitivity in %
        sensitivity = TP/float(TP+FN)*100
        
        # specificity score in %
        specificity = TN/float(TN+FP)*100
        
        # f1_score 
        f1_score = 2*(sensitivity*specificity)/(sensitivity+specificity)
        
        new_row = list(Df.loc[column])+[class_data_points, precision, sensitivity, specificity,
                                        f1_score]  # concatenate new scores in one row
        new_Df.loc[column] = new_row  # append the row to the dataframe
        
        # update initialized values
        total_data_points_number = total_data_points_number+class_data_points
        total_TP = total_TP+TP
        total_TN = total_TN+TN
        total_FN = total_FN+FN
        total_FP = total_FP+FP
    
    # after iterating throw all activity types
    # the general accuracy of the model is:
    total_accuracy = total_TP/float(total_TP+total_FN)*100
    
    # add total values to the dataframe
    new_Df.loc['Total'][additional_cols] = ['data points number='+str(total_data_points_number), '', '', '',
                                            'accuracy= '+str(total_accuracy)[0:6]+'%']
    new_Df.loc['Total'][columns] = ['' for i in range(len(columns))]
    return new_Df


# ================================================ MODEL STRUCTURES ====================================================
def build_baseline(clf_name, input_shape=(128, 6, 1), output_dim=6, lr=0.001):
    if clf_name == 'mlp':
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(window_size*2))
        model.add(Activation("relu"))
        model.add(Dropout(0.5, seed=0))
        model.add(Dense(window_size))
        model.add(Activation("relu"))
        model.add(Dropout(0.5, seed=1))
        model.add(Dense(output_dim))
        model.add(Activation("softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=lr), metrics=["accuracy"]
        )
    elif clf_name == 'cnn':
        model = Sequential()
        model.add(Conv2D(int(window_size/2), kernel_size=(5, 1), input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(Conv2D(int(window_size/2), kernel_size=(5, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(int(window_size/2), kernel_size=(5, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(int(window_size/2), kernel_size=(5, 1)))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(window_size))
        model.add(Activation("relu"))
        model.add(Dropout(0.5, seed=0))
        model.add(Dense(window_size))
        model.add(Activation("relu"))
        model.add(Dropout(0.5, seed=1))
        model.add(Dense(output_dim))
        model.add(Activation("softmax"))
        model.compile(
                loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=lr), metrics=["accuracy"]
        )
    elif clf_name == 'deep_conv_lstm':
        model = Sequential()
        model.add(Conv2D(int(window_size/2), kernel_size=(5, 1), input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(Conv2D(int(window_size/2), kernel_size=(5, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(int(window_size/2), kernel_size=(5, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(int(window_size/2), kernel_size=(5, 1)))
        model.add(Activation("relu"))
        model.add(Reshape((input_shape[0]-16, input_shape[1]*int(window_size/2))))
        model.add(LSTM(window_size, activation="tanh", return_sequences=True))
        model.add(Dropout(0.5, seed=0))
        model.add(LSTM(window_size, activation="tanh"))
        model.add(Dropout(0.5, seed=1))
        model.add(Dense(output_dim))
        model.add(Activation("softmax"))
        model.compile(
                loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=lr), metrics=["accuracy"]
        )
    else:
        models = dict()
        models['knn'] = KNeighborsClassifier(n_neighbors=7)
        models['cart'] = DecisionTreeClassifier()
        models['logReg'] = LogisticRegression()
        models['svm'] = SVC()
        models['bayes'] = GaussianNB()
        models['bag'] = BaggingClassifier(n_estimators=100)
        models['rf'] = RandomForestClassifier(n_estimators=100)
        models['et'] = ExtraTreesClassifier(n_estimators=100)
        models['gbm'] = GradientBoostingClassifier(n_estimators=100)
        models['mlpSKlearn'] = MLPClassifier()
        model = models[clf_name]
    return model


# ====================================================== TRAIN AND PREDICT FOR SINGLE MODEL ============================
def train_and_predict(LOG_DIR, key, fold_id, X_train, X_valid, y_train, y_valid, logger=None, X_test=None,
                      clf_name="", params=None, pdf=None):
    """Train CNN
    Args:
        X_train, X_valid, X_test: input signals of shape (num_samples, window_size, num_channels, 1)
        y_train, y_valid, y_test: one-hot-encoded labels
    Returns:
        pred_train: train prediction
        pred_valid: train prediction
        pred_test: train prediction
        model: trained best model
    """
    if is_NN_model(clf_name):
        model = build_baseline(
            clf_name, input_shape=X_train.shape[1:], output_dim=y_train.shape[1], lr=params["lr"])
        utils.plot_model(model, path=f"{LOG_DIR}/images/model_dataset_{key}_model_{clf_name}.pdf")
        
        csv_name = f"{LOG_DIR}/trained_model/trained_model_dataset_{key}_model_{clf_name}_fold{fold_id}"
        csv_file = open(f"{csv_name}.csv", "w")
        print("csv_name:", csv_name + ".csv")
        
        callbacks = create_callback(model=model,
                                    path_chpt=csv_name, logger=logger,
                                    verbose=5, epochs=params["epochs"])
        
        fit = model.fit(X_train, y_train, batch_size=params["batch_size"],
                        epochs=params["epochs"], verbose=params["verbose"],
                        validation_data=(X_valid, y_valid), callbacks=callbacks)
        
        utils.pickle_save(
            f"{LOG_DIR}/history_pickle/history_dataset_{key}_model_{clf_name}_fold{fold_id}.pickle",
            fit.history)
        
        utils.plot_learning_history(
            fit=fit, path=f"{LOG_DIR}/images/history_dataset_{key}_model_{clf_name}_fold{fold_id}.pdf",
            pdf_write=pdf)
        
        model.save(f"{TEST_DIR}/BestModel/trained_model_dataset_{key}_model_{clf_name}_fold{fold_id}.h5")
        model.save(f"{LOG_DIR}/trained_model/trained_model_dataset_{key}_model_{clf_name}_fold{fold_id}.h5")
    
    else:
        model = build_baseline(clf_name)
        fit = model.fit(X_train, y_train)
        joblib.dump(fit, f"{LOG_DIR}/trained_model/trained_model_dataset_{key}_model_{clf_name}_fold{fold_id}.pkl")
    
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    
    if X_test is not None: 
        pred_test = model.predict(X_test)
    else:
        pred_test = None
        
    try:
        K.clear_session()
    except:
        pass
    return pred_train, pred_valid, pred_test, model


# ====================================================== CROSS VALIDATION NO TEST=======================================
def train_cross_validate_no_test(
    clf_name, LOG_DIR, train_test_files_dic, Activities_index_name, 
    intended_dataset_number, logger, dataset_type="All", skip_fold=0):
    
    assert dataset_type == "All" or dataset_type == "last" or dataset_type in get_dataset_type(clf_name, intended_dataset_number, mode="no_test")
    
    models = []
    classifiers_summary_datasets = pd.DataFrame()
    cv_df_meta_datasets = pd.DataFrame()
    scores = initialize_scoring()
    
    dataset_type = get_dataset_type(clf_name, intended_dataset_number, mode="no_test", option="max") if dataset_type == "last" else dataset_type

    if dataset_type == 'All':
        new_dic = dict(
            filter(lambda elem: elem[0] in get_dataset_type(clf_name, intended_dataset_number, mode="no_test"), 
            train_test_files_dic.items()))
    else: 
        new_dic = {dataset_type: train_test_files_dic[dataset_type]}

    
    print(f"We will be using: {list(new_dic.keys())}")

    for key in sorted(new_dic.keys()):
        pdf = PdfPages(f"{LOG_DIR}/images/dataset_{key}_model_{clf_name}_cross_validation.pdf")
        writer = pd.ExcelWriter(f'{LOG_DIR}/cross_validation/dataset_{key}_model_{clf_name}_cv.xlsx',
                                engine='xlsxwriter')
        classifiers_summary = pd.DataFrame()
        Dataset_name = f'Dataset type {dataset_numbering[key]}'
        files = new_dic[key]
        results = {}
        
        print(f"_____________________{Dataset_name} Cross validating NO TEST______________________")
        logger.debug(f"_____________________{Dataset_name} Cross validating NO TEST______________________")
        X_train, X_test, y_train, y_test = files[0], files[1], files[2], files[3]
        
        print(f"The dataset shapes are: {X_train.shape}, {y_train.shape}")
        logger.debug(f"The dataset shapes are: {X_train.shape}, {y_train.shape}")
        
        classifiers_summary = pd.DataFrame()
        num_classes = len(np.unique(y_train))
        print(f"{clf_name} started cross-validating....")
        logger.debug(f"{clf_name} started cross-validating....")
    
        params_model = None
        try:
            with open(f"{CONFIG_DIR}/default.json", "r") as f:
                params_model = json.load(f)[f"{clf_name}_params"]
            print(f"training with params: {params_model}")
            logger.debug(params_model)
        except:
            pass

        valid_preds = np.zeros((X_train.shape[0], num_classes)) if is_NN_model(clf_name) else np.zeros(
                (X_train.shape[0]))
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=71)
        
        for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
            if fold_id >= skip_fold: 
                print(f"\n________Fold #{fold_id} started....__________")
                logger.debug(f"________Fold #{fold_id} started....__________")
                
                X_tr = X_train[train_index, :]
                X_val = X_train[valid_index, :]
                
                y_tr = y_train[train_index]
                y_val = y_train[valid_index]
    
                y_tr = keras.utils.to_categorical(y_tr, num_classes) if is_NN_model(clf_name) else y_tr
                y_val = keras.utils.to_categorical(y_val, num_classes) if is_NN_model(clf_name) else y_val
                
                if is_NN_model(clf_name):
                    pred_tr, pred_val, pred_test, model = train_and_predict(
                        LOG_DIR, key, fold_id, X_tr, X_val, y_tr, y_val, 
                        logger=logger, X_test=None, clf_name=clf_name, params=params_model, pdf=pdf)
                else:
                    pred_tr, pred_val, pred_test, model = train_and_predict(
                        LOG_DIR, key, fold_id, X_tr, X_val, y_tr, y_val, 
                        logger=logger, X_test=None, clf_name=clf_name)
    
                models.append(model)
                valid_preds[valid_index] = pred_val
                for pred, X, y, mode in zip([pred_tr, pred_val], [X_tr, X_val],
                                            [y_tr, y_val], ["train", "valid"]):
                    if is_NN_model(clf_name):
                        loss, acc = model.evaluate(X, y, verbose=0)
                        pred = pred.argmax(axis=1)
                        y = y.argmax(axis=1)
                        scores[key][mode]["log_loss"].append(loss)
                        scores[key][mode]["accuracy"].append(acc)
                    else:
                        scores[key][mode]["accuracy"].append(accuracy_score(y, pred))
                    scores[key][mode]["precision"].append(precision_score(y, pred, average="micro"))
                    scores[key][mode]["recall"].append(recall_score(y, pred, average="micro"))
                    scores[key][mode]["f1_score"].append(f1_score(y, pred, average="micro"))
                    scores[key][mode]["cm_norm"].append(cm(y, pred, normalize="true"))
                    scores[key][mode]["cm"].append(cm(y, pred))
                    scores[key][mode]["per_class_f1"].append(f1_score(y, pred, average=None))
                    scores[key]["cm_norm"][mode].append(cm(y, pred, normalize="true"))
        
        cv_df_meta = per_class_f1_df_meta = pd.DataFrame()
        
        for mode in ["train", "valid"]:
            logger.debug(f"________{mode.upper()} confusion table________")
            cv = dict((k, scores[key][mode][k]) for k in ["accuracy", "precision", "recall", "f1_score", "log_loss"])
            cv_df = pd.DataFrame.from_dict(cv, orient='index', columns=[f"Fold {i+1}" for i in range(n_splits)]).T
            cols_1 = cv_df.columns
            cv_df_meta = pd.concat([cv_df_meta, cv_df], axis=1)
            
            cv_calc = [mode]+[np.array(cv[i]).mean() for i in cv.keys()]
            cv_calc_s = pd.Series(cv_calc, index=["Mode"]+[f"Mean {i}" for i in cv.keys()])
            classifiers_summary = classifiers_summary.append(cv_calc_s, ignore_index=True)
            
            class_f1_mat = scores[key][mode]["per_class_f1"]
            class_f1_result = {}
            for class_id in range(num_classes):
                mean_class_f1 = np.mean([class_f1_mat[i][class_id] for i in range(n_splits)])
                class_f1_result[Activities_index_name[class_id]] = mean_class_f1
            per_class_f1_df = pd.DataFrame.from_dict(class_f1_result, orient='index', columns=["per_class_f1"])
            cols_2 = per_class_f1_df.columns
            per_class_f1_df_meta = pd.concat([per_class_f1_df_meta, per_class_f1_df], axis=1)
            
            columns = list(Activities_index_name.values())
            index = list(Activities_index_name.values())
            cm_array = np.mean(scores[key][mode]["cm"], axis=0)
            cm_df = pd.DataFrame(data=cm_array, columns=columns, index=index).pipe(full_confusion_matrix)
            logger.debug(cm_df.to_string().replace('\n', '\n\t'))
            cm_df.to_excel(writer, sheet_name=f'confusion_{mode}')
        
        cv_df_meta.columns = pd.MultiIndex.from_product([["train", "valid"], cols_1])
        cv_df_meta = cv_df_meta.rename_axis('Fold').reset_index()
        cv_df_meta["Classifier"] = clf_name
        cv_df_meta["Dataset"] = key
        cv_df_meta_datasets = pd.concat([cv_df_meta_datasets, cv_df_meta])

        classifiers_summary.append([scores[key]["macro averaged"]["accuracy"],
                                    scores[key]["macro averaged"]["precision"],
                                    scores[key]["macro averaged"]["recall"],
                                    scores[key]["macro averaged"]["f1_score"],
                                    "macro averaged"])
        classifiers_summary["Classifier"] = clf_name
        classifiers_summary["Dataset"] = key
        classifiers_summary_datasets = pd.concat([classifiers_summary_datasets, classifiers_summary])
        
        # Plot confusion matrix
        utils.plot_confusion_matrix(
                cms=scores[key]["cm_norm"],
                labels=list(Activities_index_name.values()),
                path=f"{LOG_DIR}/images/confusion_matrix_dataset_{key}_model_{clf_name}.pdf", pdf_write=pdf,
                clf_name=clf_name,option="no_test")
        
        np.save(f"{LOG_DIR}/valid_oof/valid_oof_dataset_{key}_model_{clf_name}.npy", valid_preds)
        logger.debug("\n\n")
        pdf.close()
        writer.save()
    
    return classifiers_summary_datasets, cv_df_meta_datasets, scores


# ====================================================== CROSS VALIDATION ==============================================
def train_cross_validate(
    clf_name, LOG_DIR, train_test_files_dic, Activities_index_name, 
    intended_dataset_number, logger, dataset_type="All", skip_fold = 0):
    
    assert dataset_type == "All" or dataset_type == "last" or dataset_type in get_dataset_type(clf_name, intended_dataset_number)
    
    models = []
    classifiers_summary_datasets = pd.DataFrame()
    cv_df_meta_datasets = pd.DataFrame()
    scores = initialize_scoring()
    
    dataset_type = get_dataset_type(clf_name, intended_dataset_number, option="max") if dataset_type == "last" else dataset_type

    if dataset_type == 'All':
        new_dic = dict(
            filter(lambda elem: elem[0] in get_dataset_type(clf_name, intended_dataset_number), 
            train_test_files_dic.items()))
    else: 
        new_dic = {dataset_type: train_test_files_dic[dataset_type]}
    
    for key in sorted(new_dic.keys()):
        pdf = PdfPages(f"{LOG_DIR}/images/dataset_{key}_model_{clf_name}_cross_validation.pdf")
        writer = pd.ExcelWriter(f'{LOG_DIR}/cross_validation/dataset_{key}_model_{clf_name}_cv.xlsx',
                                engine='xlsxwriter')
        classifiers_summary = pd.DataFrame()
        Dataset_name = f'Dataset type {dataset_numbering[key]}'
        files = new_dic[key]
        results = {}
        
        print(f"_____________________{Dataset_name} Cross validating______________________")
        logger.debug(f"_____________________{Dataset_name} Cross validating______________________")
        X_train, X_test, y_train, y_test = files[0], files[1], files[2], files[3]
        print(f"The dataset shapes are: {X_train.shape}, {y_train.shape}, {X_test.shape}, {y_test.shape}")
        logger.debug(f"The dataset shapes are: {X_train.shape}, {y_train.shape}, {X_test.shape}, {y_test.shape}")
        
        classifiers_summary = pd.DataFrame()
        num_classes = len(np.unique(y_train))
        print(f"{clf_name} started cross-validating....")
        logger.debug(f"{clf_name} started cross-validating....")
    
        params_model = None
        try:
            with open(f"{CONFIG_DIR}/default.json", "r") as f:
                params_model = json.load(f)[f"{clf_name}_params"]
            print(f"training with params: {params_model}")
            logger.debug(params_model)
        except:
            pass

        valid_preds = np.zeros((X_train.shape[0], num_classes)) if is_NN_model(clf_name) else np.zeros(
                (X_train.shape[0]))
        test_preds = np.zeros((n_splits, X_test.shape[0], num_classes)) if is_NN_model(clf_name) else np.zeros(
                (n_splits, X_test.shape[0]))
        y_test = keras.utils.to_categorical(y_test, num_classes) if is_NN_model(clf_name) else y_test
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=71)
        
        for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
            if fold_id >= skip_fold: 
                print(f"\n________Fold #{fold_id} started....__________")
                logger.debug(f"________Fold #{fold_id} started....__________")
                X_tr = X_train[train_index, :]
                X_val = X_train[valid_index, :]
                y_tr = y_train[train_index]
                y_val = y_train[valid_index]
                y_tr = keras.utils.to_categorical(y_tr, num_classes) if is_NN_model(clf_name) else y_tr
                y_val = keras.utils.to_categorical(y_val, num_classes) if is_NN_model(clf_name) else y_val
                
                if is_NN_model(clf_name):
                    pred_tr, pred_val, pred_test, model = train_and_predict(
                        LOG_DIR, key, fold_id, X_tr, X_val, y_tr, y_val,
                        logger=logger, clf_name=clf_name, X_test=X_test, params=params_model, pdf=pdf)
                else:
                    pred_tr, pred_val, pred_test, model = train_and_predict(
                        LOG_DIR, key, fold_id, X_tr, X_val, y_tr, y_val,
                        logger=logger, clf_name=clf_name, X_test=X_test)
                    
                models.append(model)
                valid_preds[valid_index] = pred_val
                test_preds[fold_id] = pred_test
                for pred, X, y, mode in zip([pred_tr, pred_val, pred_test], [X_tr, X_val, X_test],
                                            [y_tr, y_val, y_test], ["train", "valid", "test"]):
                    if is_NN_model(clf_name):
                        loss, acc = model.evaluate(X, y, verbose=0)
                        pred = pred.argmax(axis=1)
                        y = y.argmax(axis=1)
                        scores[key][mode]["log_loss"].append(loss)
                        scores[key][mode]["accuracy"].append(acc)
                    else:
                        scores[key][mode]["accuracy"].append(accuracy_score(y, pred))
                    scores[key][mode]["precision"].append(precision_score(y, pred, average="micro"))
                    scores[key][mode]["recall"].append(recall_score(y, pred, average="micro"))
                    scores[key][mode]["f1_score"].append(f1_score(y, pred, average="micro"))
                    scores[key][mode]["cm_norm"].append(cm(y, pred, normalize="true"))
                    scores[key][mode]["cm"].append(cm(y, pred))
                    scores[key][mode]["per_class_f1"].append(f1_score(y, pred, average=None))
                    scores[key]["cm_norm"][mode].append(cm(y, pred, normalize="true"))
            
        cv_df_meta = per_class_f1_df_meta = pd.DataFrame()
        
        for mode in ["train", "valid", "test"]:
            logger.debug(f"________{mode.upper()} confusion table________")
            cv = dict((k, scores[key][mode][k]) for k in ["accuracy", "precision", "recall", "f1_score", "log_loss"])
            cv_df = pd.DataFrame.from_dict(cv, orient='index', columns=[f"Fold {i+1}" for i in range(n_splits)]).T
            cols_1 = cv_df.columns
            cv_df_meta = pd.concat([cv_df_meta, cv_df], axis=1)
            
            cv_calc = [mode]+[np.array(cv[i]).mean() for i in cv.keys()]
            cv_calc_s = pd.Series(cv_calc, index=["Mode"]+["Mean "+i for i in cv.keys()])
            classifiers_summary = classifiers_summary.append(cv_calc_s, ignore_index=True)
            
            class_f1_mat = scores[key][mode]["per_class_f1"]
            class_f1_result = {}
            for class_id in range(num_classes):
                mean_class_f1 = np.mean([class_f1_mat[i][class_id] for i in range(n_splits)])
                class_f1_result[Activities_index_name[class_id]] = mean_class_f1
            per_class_f1_df = pd.DataFrame.from_dict(class_f1_result, orient='index', columns=["per_class_f1"])
            cols_2 = per_class_f1_df.columns
            per_class_f1_df_meta = pd.concat([per_class_f1_df_meta, per_class_f1_df], axis=1)
            
            columns = list(Activities_index_name.values())
            index = list(Activities_index_name.values())
            cm_array = np.mean(scores[key][mode]["cm"], axis=0)
            cm_df = pd.DataFrame(data=cm_array, columns=columns, index=index).pipe(full_confusion_matrix)
            logger.debug(cm_df.to_string().replace('\n', '\n\t'))
            cm_df.to_excel(writer, sheet_name=f'confusion_{mode}')
        
        # Output Final Scores Averaged over Folds
        
        logger.debug("________Final F1-per-class scores averaged over Folds________")
        test_pred = np.mean(test_preds, axis=0).argmax(axis=1) if is_NN_model(clf_name) else get_freq_along_axis(
                test_preds, axis=1)
        y_test_max = y_test.argmax(axis=1) if is_NN_model(clf_name) else y_test
        
        scores[key]["macro averaged"]["accuracy"] = accuracy_score(y_test_max, test_pred)
        scores[key]["macro averaged"]["precision"] = precision_score(y_test_max, test_pred, average='macro')
        scores[key]["macro averaged"]["recall"] = recall_score(y_test_max, test_pred, average='macro')
        scores[key]["macro averaged"]["f1_score"] = f1_score(y_test_max, test_pred, average='macro')
        scores[key]["macro averaged"]["per_class_f1"] = pd.DataFrame(
                f1_score(y_test_max, test_pred, average=None),
                index=per_class_f1_df_meta.index,
                columns=["per_class_f1"])
        
        per_class_f1_df_meta = pd.concat(
                [per_class_f1_df_meta, scores[key]["macro averaged"]["per_class_f1"]], axis=1)
        
        cv_df_meta.columns = pd.MultiIndex.from_product([["train", "valid", "test"], cols_1])
        per_class_f1_df_meta.columns = pd.MultiIndex.from_product([modes, cols_2])
        
        cv_df_meta = cv_df_meta.rename_axis('Fold').reset_index()
        cv_df_meta["Classifier"] = clf_name
        cv_df_meta["Dataset"] = key
        cv_df_meta_datasets = pd.concat([cv_df_meta_datasets, cv_df_meta])
        
        per_class_f1_df_meta.to_excel(writer, sheet_name=f'per_class_f1')
        logger.debug(per_class_f1_df_meta.to_string().replace('\n', '\n\t'))
        
        classifiers_summary.append([scores[key]["macro averaged"]["accuracy"],
                                    scores[key]["macro averaged"]["precision"],
                                    scores[key]["macro averaged"]["recall"],
                                    scores[key]["macro averaged"]["f1_score"],
                                    "macro averaged"])
        classifiers_summary["Classifier"] = clf_name
        classifiers_summary["Dataset"] = key
        classifiers_summary_datasets = pd.concat([classifiers_summary_datasets, classifiers_summary])
        
        # Plot confusion matrix
        utils.plot_confusion_matrix(
                cms=scores[key]["cm_norm"],
                labels=list(Activities_index_name.values()),
                path=f"{LOG_DIR}/images/confusion_matrix_dataset_{key}_model_{clf_name}.pdf", pdf_write=pdf,
                clf_name=clf_name)
        
        np.save(f"{LOG_DIR}/valid_oof/valid_oof_dataset_{key}_model_{clf_name}.npy", valid_preds)
        np.save(f"{LOG_DIR}/test_oof/test_oof_dataset_{key}_model_{clf_name}.npy",
                np.mean(test_preds, axis=0))  # Averaging
        logger.debug("\n\n")
        pdf.close()
        writer.save()
    
    return classifiers_summary_datasets, cv_df_meta_datasets, scores

# ====================================================== MAIN TRAINING TESTING functions ==============================================
def main_training_testing(model_names, train_test_files_dic, Activities_index_name, INFO, intended_dataset_number,
                          logger, dataset_type="All", option="general_method", skip_fold=0):
    
    classifiers_summary_meta = pd.DataFrame()
    cv_df_meta = pd.DataFrame()
    scores = dict()
    
    for clf_name in model_names:
        print(f"{separator} {clf_name} {separator}")
        logger.debug(f"{separator} {clf_name} {separator}")
        
        LOG_DIR = make_log_dir(INFO, clf_name=clf_name, prefix=prefix, option=option)
        utils.prepare_logs_dirs_training_testing(LOG_DIR)
        
        scores = initialize_scoring()
        scores[clf_name] = dict()
        
        mpl_logger = getLogger("matplotlib")  # Suppress matplotlib logging
        mpl_logger.setLevel(WARNING)
        
        export_mode=""
        if option=="general_method": 
            classifiers_summary, cv_df, scoring = train_cross_validate(
                clf_name, LOG_DIR, train_test_files_dic, Activities_index_name, 
                intended_dataset_number, logger, dataset_type=dataset_type, skip_fold=skip_fold)
            export_mode="test"
        else: 
            classifiers_summary, cv_df, scoring = train_cross_validate_no_test(
                clf_name, LOG_DIR, train_test_files_dic, Activities_index_name, 
                intended_dataset_number, logger, dataset_type=dataset_type, skip_fold=skip_fold)
            export_mode="no_test"

        scores[clf_name] = scoring
        classifiers_summary_meta = pd.concat([classifiers_summary_meta, classifiers_summary])
        cv_df_meta = pd.concat([cv_df_meta, cv_df])
        
        export_summary(classifiers_summary_meta, cv_df_meta, prefix, model_names, INFO, export_mode=export_mode)
    
    classifiers_summary_meta = classifiers_summary_meta.sort_values(by=["Dataset", "Mode", "Classifier"],
                                                                    ignore_index=True)
    cv_df_meta = cv_df_meta.sort_values(by=["Dataset", "Classifier", "Fold"], ignore_index=True)
    
    classifiers_summary_meta.set_index(["Dataset", "Mode", "Classifier"], inplace=True)
    cv_df_meta.set_index(["Dataset", "Classifier", "Fold"], inplace=True)
    
    logger.debug(cv_df_meta.to_string().replace('\n', '\n\t'))
    logger.debug(classifiers_summary_meta.to_string().replace('\n', '\n\t'))
    display(cv_df_meta.style.background_gradient(cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))
            .set_properties(**{'width': '65px', 'height': '20px'}))
    display(classifiers_summary_meta.style.background_gradient(cmap=sns.color_palette("flare", as_cmap=True))
            .set_properties(**{'width': '150px', 'height': '20px'}))
    
# ====================================================== LIVE STREAMING ==============================================

