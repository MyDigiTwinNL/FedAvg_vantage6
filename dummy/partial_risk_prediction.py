"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""
import pandas as pd
from typing import Any

from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.client import AlgorithmClient

from .utils import *
from .networks import DeepSurv
from .networks import NegativeLogLikelihood
from .datasets import EventDataset

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.optim as optim

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@data(1)
@algorithm_client
def partial_risk_prediction(
    client: AlgorithmClient, df1: pd.DataFrame, predictor_cols, outcome_cols, dl_config, avged_params, update_iter, n_fold, fold_index
) -> Any:

    '''
    AlgorithmClient:
    df1: the local data stored in the client node (the data never go out)
    predictor_cols: a python list of selected predictors after further data preparation step by "data_prep.py" from the harmonized FHIR data 
    outcome_cols: a python list of outcome variables after further data preparation step by "data_prep.py" from the harmonized FHIR data, it should contrain 'LENFOL'(length of follow-up) and 'FSTAT' (event occurence)
    dl_config: hyperparameter values regarding a neural network architecture and its training (specified in the configuration ini file)
    avged_params: the weighted averaged (aggregated and broadcasted by the server) model weight
    
    update_iter: the index of the current of aggregation iterations (to monitoring the progress)
    n_fold: the number of folds in the data splitting in each client for train/valid/test(set to 10 in our PoC)
    (optional) fold_index: the index of the fold that will be used for test (it is for the corrected resampled t-test in the performance evaluation)

    '''

    """ Decentral part of the algorithm """

    client_id = client.node.get()["id"]
    # print ("update_iter", update_iter)
    # print ("client_id", client_id)

    # Missing predictor data imputation by means of multiple imputation by chained equations
    imputer = IterativeImputer(random_state=0, max_iter=5)
    lenfol_col = df1['LENFOL']
    fstat_col = df1['FSTAT']
    df_for_imputation = df1.drop(columns=['LENFOL', 'FSTAT']) # exclude outcome columns (labels)
    imputer.fit(df_for_imputation)
    imputed_array = imputer.transform(df_for_imputation)
    df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)
    df_imputed['LENFOL'] = lenfol_col
    df_imputed['FSTAT'] = fstat_col


    info("Computing on client side")
    # train_df, val_df, test_df = tt_split(df_imputed)
    ## Split the data into 80%/10%/10% for training/validation/test
    train_df, val_df, test_df = ci_split(df_imputed, n_fold = n_fold, fold_index = fold_index)

    ## Vertical data split: X (feature), e (FSTAT), y(LENFOL)
    y_col = [outcome_cols[0]]
    e_col = [outcome_cols[1]]
    train_X, train_e, train_y = vertical_split(train_df, predictor_cols, y_col, e_col)
    val_X, val_e, val_y = vertical_split(val_df, predictor_cols, y_col, e_col)
    test_X, test_e, test_y = vertical_split(test_df, predictor_cols, y_col, e_col)

    ## Min-max normalization independently on each node
    train_X, X_min, X_max = normalize_train(train_X) # Normalize X
    val_X = normalize_test(val_X, X_min, X_max) # Nomralize val/test X based on min/max of train X
    test_X = normalize_test(test_X, X_min, X_max) # Nomralize val/test X based on min/max of train X

    # ## Convert numpy to torch tensor (optional)
    # train_X, train_e, train_y = np2tensor(train_X, train_e, train_y)
    # val_X, val_e, val_y = np2tensor(val_X, val_e, val_y)
    # test_X, test_e, test_y = np2tensor(test_X, test_e, test_y)

    ## Prepare dataset (PyTorch data primitive) fed into PyTorch model
    train_dataset = EventDataset(train_X, train_e, train_y)
    val_dataset = EventDataset(val_X, val_e, val_y)
    test_dataset = EventDataset(test_X, test_e, test_y)

    train_data_size = len(train_dataset)
    val_data_size = len(val_dataset)
    test_data_size = len(test_dataset)   


    batchsize = 4096

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__())

    ## Create a neural network based on the configuration specified in the ini file
    model = DeepSurv(dl_config['network']).to(device)

    # Learning rate
    learning_rate = dl_config['train']['learning_rate']
    # Load the aggregated weight received from the server 
    if avged_params is not None:
        # Use global weight by fedavg
        avged_params = json2dict(avged_params)
        model.load_state_dict(avged_params)
        learning_rate = dl_config['train']['learning_rate']/10

    # Objective function
    criterion = NegativeLogLikelihood(dl_config['network'], device).to(device)
    # Optimizer for training
    optimizer = eval('optim.{}'.format(dl_config['train']['optimizer']))(
        model.parameters(), lr=learning_rate)


    train_loss_list = []
    val_loss_list = []
    train_ci_list = []
    val_ci_list = []

    ## Training loop (the number of training epochs are also specified in the configuration file)
    for epoch in range(1, dl_config['train']['epochs']+1):
        # train step
        total_train_c = 0.0
        total_val_loss = 0
        total_train_loss = 0        
        total_train_step = 0.0
        train_total_c = 0.0
        train_total_tp = 0
        val_total_tp = 0

        model.train()
        for X, y, e in train_loader:

            X = X.to(device)
            y = y.to(device)
            e = e.to(device)
            
            risk_pred = model(X)
            risk_pred = risk_pred.to(device)

            train_loss = criterion(risk_pred, y, e, model)
            total_train_loss = total_train_loss + train_loss.item()
            train_c = c_index(-risk_pred, y, e)
            total_train_c = total_train_c + train_c
            # train_ci_list.append(train_c)
            total_train_step += 1.0
            # print ("train_c", train_c)
            # updates parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        train_loss_list.append(total_train_loss/total_train_step)
        train_ci_list.append(total_train_c/total_train_step)

        # validation step
        model.eval()
        with torch.no_grad():
            for X, y, e in val_loader:

                X = X.to(device, dtype=float)
                y = y.to(device, dtype=float)
                e = e.to(device, dtype=float)

                risk_pred = model(X)
                risk_pred = risk_pred.to(device)
                valid_loss = criterion(risk_pred, y, e, model)
                valid_c = c_index(-risk_pred, y, e)
                val_ci_list.append(valid_c)
                val_loss_list.append(valid_loss.item())


    ## Plot training curves
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figure_temp_dir = os.path.join(current_dir, "figure_ci_%s" %fold_index )
    if not os.path.exists(figure_temp_dir):
        os.makedirs(figure_temp_dir)
    train_log={"train_c-index": train_ci_list, "val_c-index": val_ci_list, "train_loss": train_loss_list, "val_loss": val_loss_list}
    plot_train_curve_ci(train_log, figure_temp_dir, update_iter, client_id)

    ## Test trained model and save 
    test_cm_dict = {}
    test_eval_dict = {}
    output_list = []
    label_list = []
    pred_class_list =[]

    test_total_tp = 0

    ## Test
    model.eval()
    with torch.no_grad():
        for X, y, e in test_loader:

            X = X.to(device, dtype=float)
            y = y.to(device, dtype=float)
            e = e.to(device, dtype=float)
            risk_pred = model(X)
            risk_pred = risk_pred.to(device)
            test_c = c_index(-risk_pred, y, e)

    test_cm_dict['risk_pred'] = risk_pred.numpy().tolist()
    test_cm_dict['y'] = y.numpy().tolist()
    test_cm_dict['e'] = e.numpy().tolist()
    test_eval_dict['ci'] = test_c.tolist()

    
    ## Return client's weights (after local training)
    ## See ttps://github.com/itslastonenikhil/federated-learning/blob/main/FederatedLearning.ipynb
    model_params = model.state_dict()
    for entry in model_params:
        model_params[entry] = model_params[entry].cpu().data.numpy().tolist()

    model_params_json = json.dumps(model_params)    
    test_cm_dict = json.dumps(test_cm_dict)
    test_eval_dict = json.dumps(test_eval_dict)

    # Return results to the vantage6 server.
    return {"params": model_params_json, "num_train_samples": train_data_size, "test_cm": test_cm_dict, "test_eval": test_eval_dict}

