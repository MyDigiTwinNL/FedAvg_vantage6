'''
Python script for common functions for Partials and performance evaluation.
'''
import os
import time
import json
import logging
import random
import numpy as np
import torch
import configparser
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from vantage6.algorithm.tools.util import info, warn, error

from lifelines.utils import concordance_index


torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
random.seed(0)
np.random.seed(0)

def read_config(ini_file):
    ''' Performs read config file and parses it.

    :param ini_file: (String) the path of a .ini file.
    :return config: (dict) the dictionary of information in ini_file.
    '''
    def _build_dict(items):
        return {item[0]: eval(item[1]) for item in items}
    # create configparser object
    cf = configparser.ConfigParser()
    # read .ini file
    cf.read(ini_file)
    config = {sec: _build_dict(cf.items(sec)) for sec in cf.sections()}
    return config

def c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''

    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
        
    return concordance_index(y, risk_pred, e)

def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate):
    ''' Adjusts learning rate according to (epoch, lr and lr_decay_rate)

    :param optimizer: (torch.optim object)
    :param epoch: (int)
    :param lr: (float) the initial learning rate
    :param lr_decay_rate: (float) learning rate decay rate
    :return lr_: (float) updated learning rate
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1+epoch*lr_decay_rate)
    return optimizer.param_groups[0]['lr']

def create_logger(logs_dir):
    ''' Performs creating logger

    :param logs_dir: (String) the path of logs
    :return logger: (logging object)
    '''
    # logs settings
    log_file = os.path.join(logs_dir,
                            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.log')

    # initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # initialize handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # initialize console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # builds logger
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def tt_split(df_input, stratify_colname='FSTAT',
                                         frac_train=0.6, frac_val=0.2, frac_test=0.2,
                                         random_state=0):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)



    return df_train, df_val, df_test


def ci_split(df_input, stratify_colname='FSTAT', n_fold = 10, 
                                         frac_train=0.8, frac_val=0.1, frac_test=0.1,
                                         random_state=0, fold_index=0):
    '''
    StratifiedKFold for confidence interval
    '''
    info("ci_split step 1- StratifiedKFold")
    skf = StratifiedKFold(n_splits=n_fold)

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.
    
    info("ci_split step 2")
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        # print(f"Fold {i}:")
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")
        # print ("len(train_index)", len(train_index))
        # print ("len(test_index)", len(test_index))
        if i == fold_index:
            selected_k_train_index = train_index
            selected_k_test_index = test_index

    info("ci_split step 3")
    df_train = X.loc[selected_k_train_index]
    df_test = X.loc[selected_k_test_index]
    y_train = y.loc[selected_k_train_index]
    y_test = y.loc[selected_k_test_index]

    info("ci_split step 4")
    # Split original dataframe into train and temp dataframes.
    relative_frac_test = len(y_test)/len(y_train)
    df_train, df_val, y_train, y_val = train_test_split(df_train,
                                                      y_train,
                                                      stratify=y_train,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    # assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test



def vertical_split(df, predictor_cols, y_col, e_col):

    ''' The function to parsing data from df.

    :return X: (np.array) (n, m)
        m is features dimension.
    :return e: (np.array) (n, 1)
        whether the event occurs? (1: occurs; 0: others)
    :return y: (np.array) (n, 1)
        the time of event e.
    '''
    X = df[predictor_cols].values
    e = df[e_col].values.reshape(-1, 1)
    y = df[y_col].values.reshape(-1, 1)

    return X, e, y


def normalize_train(train_X):
    X_min = train_X.min(axis=0)
    X_max = train_X.max(axis=0)
    
    # Handle zero-range features
    zero_range_mask = (X_max == X_min)
    X_max[zero_range_mask] += 1e-8  # Add a small epsilon
    
    normalized_train_x = (train_X - X_min) / (X_max - X_min)
    
    # Replace NaN values with 0
    normalized_train_x[np.isnan(normalized_train_x)] = 0
    
    return normalized_train_x, X_min, X_max


# def normalize_train(train_X):

#     X_min = train_X.min(axis=0)
#     X_max = train_X.max(axis=0)

#     if (X_min==X_max).all(): # For dummy data that have the same value for all the columns
#         X_max = X_max/2  # set as the half of the max values to avoid zero division.
#         X_min = X_max/2  # set as the half of the max values to avoid zero division.
#         normalized_train_x = (train_X-X_min) / (X_max-X_min)
#         # convert nan col (that have zeros only) to zero
#         normalized_train_x[np.isnan(normalized_train_x)] = 0

#     else:
#         normalized_train_x = (train_X-X_min) / (X_max-X_min)

#         #Workaround
#         normalized_train_x = np.nan_to_num(normalized_train_x)


#     return normalized_train_x, X_min, X_max



def normalize_test(test_X, X_min, X_max):

    normalized_test_x = (test_X-X_min) / (X_max-X_min)
    normalized_test_x[np.isnan(normalized_test_x)] = 0

    return normalized_test_x


def np2tensor(X_array, e_array, y_array):
    X_tensor = torch.from_numpy(X_array)
    e_tensor = torch.from_numpy(e_array)
    y_tensor = torch.from_numpy(y_array)
    return X_tensor, e_tensor, y_tensor


def fed_avg(params_list, num_train_samples_list):
    avged_params = params_list[0]
    for key in params_list[0]:
        weighted_sum = np.zeros(avged_params[key].shape)
        total_num = 0
        for i in range(len(params_list)):
            weighted_sum += num_train_samples_list[i] * params_list[i][key] 
            total_num += num_train_samples_list[i]
        avged_params[key] = torch.from_numpy(weighted_sum / total_num)
    return avged_params

def dict2json(s_dict):
    for entry in s_dict:
        s_dict[entry] = s_dict[entry].cpu().data.numpy().tolist()

    model_params_json = json.dumps(s_dict)

    return model_params_json


def json2dict(model_params_json):
    params = json.loads(model_params_json) 
    for entry in params:
        params[entry] = torch.from_numpy(np.array(params[entry]))
 
    return params


def plot_train_curve(train_log, figure_temp_dir, update_iter, client_id):

    ## Save Training & validation curve
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax = ax.ravel()

    for i, metric in enumerate(["acc", "loss"]):
        ax[i].plot(train_log["train_" + metric])
        ax[i].plot(train_log["val_" + metric])
        ax[i].set_title("Model {}".format(metric), fontsize=16)
        ax[i].set_xlabel("epochs", fontsize=16)
        ax[i].set_ylabel(metric, fontsize=16)

        ax[i].tick_params(axis="x", labelsize=14)
        ax[i].tick_params(axis="y", labelsize=14)

        ax[i].legend(["train", "val"], fontsize=16)

    plt.savefig(os.path.join(figure_temp_dir, 'training_curves_iter-%s_client-%s.png' %(update_iter, client_id) ), bbox_inches='tight')
    # plt.savefig(os.path.join(figure_temp_dir, 'training_curves_iter-%s_client-%s.eps' %(update_iter, client_id) ), bbox_inches='tight')
    plt.close()

    return

def plot_train_curve_ci(train_log, figure_temp_dir, update_iter, client_id):

    ## Save Training & validation curve
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax = ax.ravel()

    for i, metric in enumerate(["c-index", "loss"]):
        ax[i].plot(train_log["train_" + metric])
        ax[i].plot(train_log["val_" + metric])
        ax[i].set_title("Model {}".format(metric), fontsize=16)
        ax[i].set_xlabel("epochs", fontsize=16)
        ax[i].set_ylabel(metric, fontsize=16)

        ax[i].tick_params(axis="x", labelsize=14)
        ax[i].tick_params(axis="y", labelsize=14)

        ax[i].legend(["train", "val"], fontsize=16)

    plt.savefig(os.path.join(figure_temp_dir, 'training_curves_ci_iter-%s_client-%s.png' %(update_iter, client_id) ), bbox_inches='tight')
    # plt.savefig(os.path.join(figure_temp_dir, 'training_curves_iter-%s_client-%s.eps' %(update_iter, client_id) ), bbox_inches='tight')
    plt.close()

    return

# def plot_global_results(glo_result_list, figure_result_dir, metric):

#     xdata = range(len(glo_result_list)-1)
#     ydata = glo_result_list[1:]


#     # fig = plt.figure(figsize=(7.2, 4.2))
#     fig = plt.figure()

#     plt.plot(xdata, ydata, 'go-', label = "FedAvg")
#     plt.axhline(y= glo_result_list[0], color='r', linestyle='--' , label = "Before aggregation")
#     plt.legend(fontsize=10)
#     plt.ylabel("Global %s" %metric, fontsize=15)
#     plt.xlabel("Update iteration", fontsize=15)
#     plt.xticks(xdata, range(1, len(glo_result_list)), fontsize=13, rotation=70 )
#     plt.yticks(fontsize=13)
#     plt.title("Global %s across update iteration" %metric)
#     # plt.ylim([0,500])

#     fig.savefig(os.path.join(figure_result_dir, 'global_eval_%s.png' %metric), dpi=500 ,bbox_inches='tight')
#     # fig.savefig(os.path.join(figure_result_dir, 'global_eval_%s.png' %metric), dpi=1500 ,bbox_inches='tight')


#     plt.close()

#     return


def plot_global_results(glo_result_list, figure_result_dir, metric):

    xdata = range(len(glo_result_list)-1)
    ydata = glo_result_list[1:]


    # fig = plt.figure(figsize=(7.2, 4.2))
    fig = plt.figure(figsize=(4,3))

    plt.plot(xdata, ydata, 'go-', label = "FedAvg")
    plt.axhline(y= glo_result_list[0], color='r', linestyle='--' , label = "Before aggregation")
    plt.legend(loc = "upper left", fontsize=11)
    plt.ylabel("Global %s" %metric, fontsize=15)
    plt.xlabel("Update iteration", fontsize=15)
    plt.xticks(xdata, range(1, len(glo_result_list)), fontsize=12, rotation=70 )
    plt.yticks(fontsize=12)
    # plt.title("Global %s across update iteration" %metric)
    # plt.ylim([0,500])

    fig.savefig(os.path.join(figure_result_dir, 'global_eval_%s_lifelines.png' %metric), dpi=500 ,bbox_inches='tight')
    fig.savefig(os.path.join(figure_result_dir, 'global_eval_%s_lifelines.eps' %metric), dpi=500 ,bbox_inches='tight')


    plt.close()

    return


def plot_local_results(local_result_list, figure_result_dir, metric):

    xdata = range(len(local_result_list)-1)
    ydata = []
    for i in range(len(local_result_list[0])):
        ydata.append([])

    for loc_result in  local_result_list:
        for cli_i, cli_result in enumerate(loc_result):
            ydata[cli_i].append(cli_result)


    # fig = plt.figure(figsize=(7.2, 4.2))
    fig = plt.figure(figsize=(4,3))
    c_list = ['go-', 'ro-', 'bo-']
    c_hl_list = ['g', 'r', 'b']
    for index, c_i, c_hl in zip(range(len(ydata)), c_list, c_hl_list):
        plt.plot(xdata, ydata[index][1:], c_i, label = 'FedAvg, client %s' %(index+1))
        plt.axhline(y= ydata[index][0], color=c_hl, linestyle='--' , label = "Before aggr., client %s " %(index+1))

    plt.legend(fontsize=8)
    plt.ylabel("Local %s" %metric, fontsize=15)
    plt.xlabel("Update iteration", fontsize=15)
    plt.xticks(xdata, range(1, len(local_result_list)), fontsize=12, rotation=70 )
    plt.yticks(fontsize=12)
    # plt.title("Local %s across update iteration" %metric)
    # plt.ylim([0,500])

    fig.savefig(os.path.join(figure_result_dir, 'local_eval_%s_lifelines.png' %metric), dpi=500 ,bbox_inches='tight')
    fig.savefig(os.path.join(figure_result_dir, 'local_eval_%s_lifelines.eps' %metric), dpi=500 ,bbox_inches='tight')


    plt.close()

    return