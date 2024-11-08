'''
Python script for common function for partial tasks
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


## train_test_split 
def train_test_split(df, train_raio=0.6, random_state=0):

    train_df = df.sample(frac = train_raio, random_state=random_state)
    df_eval = df.drop(train_df.index)
    val_df = df_eval.sample(frac = 0.5, random_state=random_state)
    test_df = df_eval.drop(val_df.index)


    return train_df, val_df, test_df


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
    normalized_train_x = (train_X-X_min) / (X_max-X_min)
    return normalized_train_x, X_min, X_max

def normalize_test(test_X, X_min, X_max):
    normalized_test_x = (test_X-X_min) / (X_max-X_min)
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

def plot_global_results(glo_result_list, figure_result_dir, metric):

    xdata = range(len(glo_result_list))
    ydata = glo_result_list


    # fig = plt.figure(figsize=(7.2, 4.2))
    fig = plt.figure()

    plt.plot(xdata, ydata, 'go-')
    # plt.legend(fontsize=10)
    plt.ylabel("Global %s" %metric, fontsize=15)
    plt.xlabel("Update iteration", fontsize=15)
    plt.xticks(xdata, range(1, len(glo_result_list)+1), fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("Global %s across update iteration" %metric)
    # plt.ylim([0,500])

    fig.savefig(os.path.join(figure_result_dir, 'global_eval_%s.png' %metric), dpi=500 ,bbox_inches='tight')
    # fig.savefig(os.path.join(figure_result_dir, 'global_eval_%s.png' %metric), dpi=1500 ,bbox_inches='tight')


    plt.close()

    return


def plot_local_results(local_result_list, figure_result_dir, metric):

    xdata = range(len(local_result_list))
    ydata = []
    for i in range(len(local_result_list[0])):
        ydata.append([])

    for loc_result in  local_result_list:
        for cli_i, cli_result in enumerate(loc_result):
            ydata[cli_i].append(cli_result)


    # fig = plt.figure(figsize=(7.2, 4.2))
    fig = plt.figure()
    c_list = ['go-', 'ro-', 'bo-']
    for index, c_i in zip(range(len(ydata)), c_list):
        plt.plot(xdata, ydata[index], c_i, label = 'Client %s' %(index+1))

    plt.legend(fontsize=10)
    plt.ylabel("Local %s" %metric, fontsize=15)
    plt.xlabel("Update iteration", fontsize=15)
    plt.xticks(xdata, range(1, len(local_result_list)+1), fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("Local %s across update iteration" %metric)
    # plt.ylim([0,500])

    fig.savefig(os.path.join(figure_result_dir, 'local_eval_%s.png' %metric), dpi=500 ,bbox_inches='tight')
    # fig.savefig(os.path.join(figure_result_dir, 'global_eval_%s.png' %metric), dpi=1500 ,bbox_inches='tight')


    plt.close()

    return