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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@data(1)
@algorithm_client
def partial(
    client: AlgorithmClient, df1: pd.DataFrame, predictor_cols, outcome_cols, dl_config
) -> Any:

    """ Decentral part of the algorithm """

    info("Computing on client side")
    train_df, val_df, test_df = train_test_split(df1)
    print (train_df)
    print ("dl_config", dl_config)

    ## Vertical data split: X (feature), e (FSTAT), y(LENFOL)
    y_col = [outcome_cols[0]]
    e_col = [outcome_cols[1]]

    train_X, train_e, train_y = vertical_split(train_df, predictor_cols, y_col, e_col)
    val_X, val_e, val_y = vertical_split(val_df, predictor_cols, y_col, e_col)
    test_X, test_e, test_y = vertical_split(test_df, predictor_cols, y_col, e_col)

    train_X, X_min, X_max = normalize_train(train_X) # Normalize X
    val_X = normalize_test(val_X, X_min, X_max) # Nomralize val/test X based on min/max of train X
    test_X = normalize_test(test_X, X_min, X_max) # Nomralize val/test X based on min/max of train X

    # ## Convert numpy to torch tensor
    # train_X, train_e, train_y = np2tensor(train_X, train_e, train_y)
    # val_X, val_e, val_y = np2tensor(val_X, val_e, val_y)
    # test_X, test_e, test_y = np2tensor(test_X, test_e, test_y)

    train_dataset = EventDataset(train_X, train_e, train_y)
    val_dataset = EventDataset(val_X, val_e, val_y)
    test_dataset = EventDataset(test_X, test_e, test_y)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_dataset.__len__())

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__())

    model = DeepSurv(config['network']).to(device)
    criterion = CrossEntropyLoss()
    # criterion = NegativeLogLikelihood(config['network'], device).to(device)
    optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
        model.parameters(), lr=config['train']['learning_rate'])

    for epoch in range(1, config['train']['epochs']+1):
        # train step
        model.train()
        for X, y, e in train_loader:
            # makes predictions
            print ("device", device)
            X = X.to(device)
            y = y.to(device)
            e = e.to(device)
            
            risk_pred = model(X)
            risk_pred = risk_pred.to(device)

            train_loss = criterion(risk_pred, y, e, model)
            train_c = c_index(-risk_pred, y, e)
            # updates parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

    # # print (train_X.shape)
    # # print (train_e.shape)
    # # print (train_y.shape)

    # ## Dataset preparation

    # ## Generate neural network

    # model = fp_classifier.VGG_Net(in_channel = train_x.shape[1])
    # optimizer = Adam(model.parameters(), lr=learning_rate)
    # criterion = CrossEntropyLoss()


    # # Return results to the vantage6 server.
    # # return {"count": float(count), "sum":float(sum_)}

# TODO Feel free to add more partial functions here.
