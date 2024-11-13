"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""
import pandas as pd
import json
from typing import Any
import random
import numpy as np
from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.client import AlgorithmClient
from sklearn.metrics import accuracy_score, roc_curve, auc, recall_score
from sklearn.metrics import confusion_matrix
from .utils import *
from .networks import DeepSurv
from .networks import NegativeLogLikelihood
from .datasets import EventDataset

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
random.seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.set_default_dtype(float)

@data(1)
@algorithm_client
def partial(
    client: AlgorithmClient, df1: pd.DataFrame, predictor_cols, outcome_cols, dl_config, avged_params, update_iter
) -> Any:

    """ Decentral part of the algorithm """


    # print ("client.node.get()", client.node.get() )
    client_id = client.node.get()["id"]
    print ("update_iter", update_iter)
    print ("client_id", client_id)

    info("Computing on client side")
    # train_df, val_df, test_df = tt_split(df1)
    train_df, val_df, test_df = tt_split(df1)
    # print (train_df)
    # print ("dl_config", dl_config)

    ## Vertical data split: X (feature), e (FSTAT), y(LENFOL)
    y_col = [outcome_cols[0]]
    e_col = [outcome_cols[1]]

    train_X, train_e, train_y = vertical_split(train_df, predictor_cols, y_col, e_col)
    val_X, val_e, val_y = vertical_split(val_df, predictor_cols, y_col, e_col)
    test_X, test_e, test_y = vertical_split(test_df, predictor_cols, y_col, e_col)

    train_X, X_min, X_max = normalize_train(train_X) # Normalize X
    val_X = normalize_test(val_X, X_min, X_max) # Nomralize val/test X based on min/max of train X
    test_X = normalize_test(test_X, X_min, X_max) # Nomralize val/test X based on min/max of train X

    train_dataset = EventDataset(train_X, train_e, train_y)
    val_dataset = EventDataset(val_X, val_e, val_y)
    test_dataset = EventDataset(test_X, test_e, test_y)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__())

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=32)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=32)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=32)

    train_data_size = len(train_dataset)
    val_data_size = len(val_dataset)
    test_data_size = len(test_dataset)   

    print ("train_data_size", train_data_size)
    print ("val_data_size", val_data_size)
    print ("test_data_size", test_data_size)

    model = DeepSurv(dl_config['network']).to(device, dtype=float)
    learning_rate = dl_config['train']['learning_rate']
    if avged_params is not None:
        # Use global weight by fedavg
        avged_params = json2dict(avged_params)
        model.load_state_dict(avged_params)
        learning_rate = dl_config['train']['learning_rate']/10



    criterion = CrossEntropyLoss()
    # print ("learning_rate", learning_rate)
    # criterion = NegativeLogLikelihood(config['network'], device).to(device)
    optimizer = eval('optim.{}'.format(dl_config['train']['optimizer']))(
        model.parameters(), lr=learning_rate)

    # print ("model", model)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(1, dl_config['train']['epochs']+1):
        # train step
        model.train()

        total_test_loss = 0
        total_trainaccuracy = 0
        total_testaccuracy = 0
        total_val_loss = 0
        total_train_loss = 0        
        total_train_step = 0        
        train_total_tp = 0
        val_total_tp = 0

        # print("epoch", epoch)

        for X, y, e in train_loader:
            # makes predictions
            X = X.to(device, dtype=float)
            y = y.to(device, dtype=float)

            # e = e.to(device, dtype=float)
            e = e.flatten()
            e = e.type(torch.LongTensor) 
            e = e.to(device)

            # event_pred = model(X)
            outputs = model(X)
            targets = e

            exp = torch.exp(outputs).cpu()
            exp_sum = torch.sum(exp, dim=1) 
            softmax = exp/exp_sum.unsqueeze(-1)            
            prob = list(softmax.detach().cpu().numpy())

            # print ("prob", prob)
            predictions = np.argmax(prob, axis=1)
            targets_array = targets.detach().cpu().numpy().astype(int)


            tp_count = (predictions == targets_array).sum()
            train_total_tp += tp_count
            # print ("tp_count", tp_count)

            loss = criterion(outputs, targets)
            total_train_loss = total_train_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print ("loss: ",  loss.item())


        model.eval()
        with torch.no_grad():
            for X, y, e in val_loader:

                X = X.to(device, dtype=float)
                y = y.to(device, dtype=float)

                # e = e.to(device, dtype=float)
                e = e.flatten()
                e = e.type(torch.LongTensor) 
                e = e.to(device)

                # event_pred = model(X)
                outputs = model(X)
                targets = e

                exp = torch.exp(outputs).cpu()
                exp_sum = torch.sum(exp, dim=1) 
                softmax = exp/exp_sum.unsqueeze(-1)
                prob = list(softmax.detach().cpu().numpy())                

                predictions = np.argmax(prob, axis=1)
                targets_array = targets.detach().cpu().numpy().astype(int)
                tp_count = (predictions == targets_array).sum()
                val_total_tp += tp_count


                val_loss = criterion(outputs, targets)
                total_val_loss = total_val_loss + val_loss.item()


        train_acc = train_total_tp / train_data_size
        val_acc = val_total_tp / val_data_size
        # print("train accuracy: {}".format(train_acc))
        # print("val accuracy: {}".format(val_acc))

        train_loss_list.append(total_train_loss)
        val_loss_list.append(total_val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    ## Plot training curves
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figure_temp_dir = os.path.join(current_dir, "figure")
    if not os.path.exists(figure_temp_dir):
        os.makedirs(figure_temp_dir)

    train_log={"train_acc": train_acc_list, "val_acc": val_acc_list, "train_loss": train_loss_list, "val_loss": val_loss_list}
    plot_train_curve(train_log, figure_temp_dir, update_iter, client_id)


    ## Test trained model and save 
    test_cm_dict = {}
    test_eval_dict = {}
    test_pred_dict = {}

    output_list = []
    label_list = []
    pred_class_list =[]

    test_total_tp = 0

    model.eval()
    with torch.no_grad():
        for X, y, e in test_loader:

            X = X.to(device, dtype=float)
            y = y.to(device, dtype=float)

            # e = e.to(device, dtype=float)
            e = e.flatten()
            e = e.type(torch.LongTensor) 
            e = e.to(device)

            # event_pred = model(X)
            outputs = model(X)
            targets = e

            exp = torch.exp(outputs).cpu()
            exp_sum = torch.sum(exp, dim=1) 
            softmax = exp/exp_sum.unsqueeze(-1)
            prob = list(softmax.detach().cpu().numpy())                

            predictions = np.argmax(prob, axis=1)
            targets_array = targets.detach().cpu().numpy().astype(int)
            tp_count = (predictions == targets_array).sum()
            test_total_tp += tp_count

            for item in prob:
                output_list.append(item[1])

        test_acc = test_total_tp / test_data_size


    y_test = targets_array
    test_result = predictions
    prediction_results = output_list

    # fpr, tpr, thresholds = roc_curve(y_test, test_result, pos_label=2)
    fpr, tpr, thresholds = roc_curve(y_test, prediction_results)
    auc_value = auc(fpr, tpr)
    # print ("auc_value", auc_value)
    # print ("test_acc", test_acc)

    tn, fp, fn, tp = confusion_matrix(y_test, test_result).ravel()
    accuracy = (tp+tn) / (tp+fp+fn+tn)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn) 

    test_cm_dict['tn'] = int(tn)
    test_cm_dict['fp'] = int(fp)
    test_cm_dict['fn'] = int(fn)
    test_cm_dict['tp'] = int(tp)

    test_eval_dict['accuracy'] = accuracy
    test_eval_dict['specificity'] = specificity
    test_eval_dict['sensitivity'] = sensitivity
    test_eval_dict['auc_value'] = auc_value

    test_pred_dict['y_test'] = y_test.tolist()
    test_pred_dict['prediction_results'] = prediction_results

    ## return client's weights (after local training)
    ## https://github.com/itslastonenikhil/federated-learning/blob/main/FederatedLearning.ipynb
    model_params = model.state_dict()

    for entry in model_params:
        model_params[entry] = model_params[entry].cpu().data.numpy().tolist()


    model_params_json = json.dumps(model_params)
    
    test_cm_dict = json.dumps(test_cm_dict)
    test_eval_dict = json.dumps(test_eval_dict)
    test_pred_dict = json.dumps(test_pred_dict)



    # Return results to the vantage6 server.
    return {"params": model_params_json, "num_train_samples": train_data_size, "test_cm": test_cm_dict, "test_eval": test_eval_dict, "test_pred": test_pred_dict}

# TODO Feel free to add more partial functions here.
