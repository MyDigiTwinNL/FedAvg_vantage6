"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""
import pandas as pd
import numpy as np
from typing import Any

import vantage6

from vantage6.algorithm.tools.util import info, warn, error


# Vantage6 decorator imports (compatible across versions)
try:
    from vantage6.algorithm.tools.decorators import algorithm_client, data
except ModuleNotFoundError:
    from vantage6.algorithm.decorators import algorithm_client, data

from vantage6.algorithm.client import AlgorithmClient

from importlib import metadata as md

def log_vantage_dists(info_fn):
    items = []
    for d in md.distributions():
        name = (d.metadata.get("Name") or "").strip()
        if "vantage" in name.lower():
            items.append(f"{name}=={d.version}")
    info_fn("[pkg-check] " + (", ".join(sorted(items)) if items else "NO vantage* distributions found"))



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
from sklearn.impute import IterativeImputer, SimpleImputer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd

def drop_duplicated_header_rows_strict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strictly drop only rows that are duplicated CSV headers.
    A duplicated header row typically has values equal to the column names.
    This version is designed to avoid removing any legitimate data rows.
    """
    if df is None or df.empty:
        return df

    cols = list(df.columns)
    if not cols:
        return df

    # Compare each cell to its column name (string-wise)
    colname_row = pd.Series({c: str(c).strip() for c in cols})
    mask_full = df[cols].astype(str).apply(lambda s: s.str.strip()).eq(colname_row, axis=1).all(axis=1)

    # Extra-safe: also catch partial header rows by checking key outcome columns if present
    key_cols = [c for c in ["FSTAT", "LENFOL"] if c in df.columns]
    if key_cols:
        mask_key = df[key_cols].astype(str).apply(lambda s: s.str.strip()).eq(
            pd.Series({c: c for c in key_cols}), axis=1
        ).all(axis=1)
        mask = mask_full | mask_key
    else:
        mask = mask_full

    if mask.any():
        df = df.loc[~mask].copy()
        df.reset_index(drop=True, inplace=True)

    return df



def _sanitize_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make outcomes safe for StratifiedKFold:
      - FSTAT must be numeric and binary {0,1}
      - LENFOL must be numeric
      - Drop rows with missing outcomes
    """
    df = df.copy()

    if "FSTAT" not in df.columns or "LENFOL" not in df.columns:
        raise ValueError("Expected outcome columns FSTAT and LENFOL in dataframe")

    # Coerce to numeric (mixed types -> NaN)
    df["FSTAT"]  = pd.to_numeric(df["FSTAT"], errors="coerce")
    df["LENFOL"] = pd.to_numeric(df["LENFOL"], errors="coerce")

    # Drop missing outcomes
    n_before = len(df)
    df = df.dropna(subset=["FSTAT", "LENFOL"]).copy()
    n_after = len(df)

    # Force binary event indicator
    # (handles {0,1}, {1,2}, booleans, floats, etc.)
    df["FSTAT"] = (df["FSTAT"] > 0).astype(int)

    # Log quick diagnostics
    vc = df["FSTAT"].value_counts(dropna=False).to_dict()
    info(f"[sanitize] dropped {n_before - n_after} rows with NaN outcomes; FSTAT counts={vc}")

    # Must have 2 classes for stratification
    if df["FSTAT"].nunique() < 2:
        warn(f"[sanitize] FSTAT has <2 classes after cleaning: {vc}. Stratified split will fail.")
        # We don't raise here; Option B will fall back to non-stratified split.
    return df


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

    log_vantage_dists(info)



    # client_id = client.node.get()["id"]
    client_id = client.organization_id
    # print ("update_iter", update_iter)
    # print ("client_id", client_id)
    info(f"client_id {client_id} ")

    df1 = drop_duplicated_header_rows_strict(df1)
    df1 = _sanitize_outcomes(df1)

    # Missing predictor data imputation by means of multiple imputation by chained equations
    # imputer = IterativeImputer(random_state=0, max_iter=5, keep_empty_features=True) # keep empty features to eschew errors during the development (if the array include dummy columns)
    imputer = SimpleImputer(strategy="median") # keep empty features to eschew errors during the development (if the array include dummy columns)
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

    #info("Vertical data split")
    ## Vertical data split: X (feature), e (FSTAT), y(LENFOL)
    y_col = [outcome_cols[0]]
    e_col = [outcome_cols[1]]
    train_X, train_e, train_y = vertical_split(train_df, predictor_cols, y_col, e_col)
    val_X, val_e, val_y = vertical_split(val_df, predictor_cols, y_col, e_col)
    test_X, test_e, test_y = vertical_split(test_df, predictor_cols, y_col, e_col)

    #info("Min-max normalization")
    ## Min-max normalization independently on each node
    train_X, X_min, X_max = normalize_train(train_X) # Normalize X
    val_X = normalize_test(val_X, X_min, X_max) # Nomralize val/test X based on min/max of train X
    test_X = normalize_test(test_X, X_min, X_max) # Nomralize val/test X based on min/max of train X

    # ## Convert numpy to torch tensor (optional)
    # train_X, train_e, train_y = np2tensor(train_X, train_e, train_y)
    # val_X, val_e, val_y = np2tensor(val_X, val_e, val_y)
    # test_X, test_e, test_y = np2tensor(test_X, test_e, test_y)

    ## Eschew OOM
    train_X = train_X.astype("float32", copy=False)
    val_X   = val_X.astype("float32", copy=False)
    test_X  = test_X.astype("float32", copy=False)

    train_y = train_y.astype("float32", copy=False)
    val_y   = val_y.astype("float32", copy=False)
    test_y  = test_y.astype("float32", copy=False)

    train_e = train_e.astype("float32", copy=False)
    val_e   = val_e.astype("float32", copy=False)
    test_e  = test_e.astype("float32", copy=False)

    ## Prepare dataset (PyTorch data primitive) fed into PyTorch model
    train_dataset = EventDataset(train_X, train_e, train_y)
    val_dataset = EventDataset(val_X, val_e, val_y)
    test_dataset = EventDataset(test_X, test_e, test_y)

    train_data_size = len(train_dataset)
    val_data_size = len(val_dataset)
    test_data_size = len(test_dataset)   


    # batchsize = 4096
    batchsize = 1024
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__())


    #info("Create a neural network based on the configuration specified in the ini file")
    ## Create a neural network based on the configuration specified in the ini file

    # model = DeepSurv(dl_config['network']).to(device)

    # # Learning rate
    # learning_rate = dl_config['train']['learning_rate']
    # # Load the aggregated weight received from the server 
    # if avged_params is not None:
    #     # Use global weight by fedavg
    #     avged_params = json2dict(avged_params)
    #     model.load_state_dict(avged_params)
    #     learning_rate = dl_config['train']['learning_rate']/10


    model = DeepSurv(dl_config['network']).to(device).float()  #  force float32

    learning_rate = dl_config['train']['learning_rate']

        
    if avged_params is not None:
        avged_params = json2dict(avged_params)

        #  convert numpy/list -> torch.Tensor(float32) for load_state_dict
        state = {}
        for k, v in avged_params.items():
            # v can be list or np.ndarray
            arr = np.asarray(v, dtype=np.float32)
            state[k] = torch.from_numpy(arr)

        missing, unexpected = model.load_state_dict(state, strict=False)
        info(f"[load_state] missing={missing}, unexpected={unexpected}")

        #  keep model float32
        model = model.float()

        learning_rate = dl_config["train"]["learning_rate"] / 10


        

    #info("Objective function")
    # Objective function
    criterion = NegativeLogLikelihood(dl_config['network'], device).to(device)
    # Optimizer for training
    optimizer = eval('optim.{}'.format(dl_config['train']['optimizer']))(
        model.parameters(), lr=learning_rate)


    train_loss_list = []
    val_loss_list = []
    train_ci_list = []
    val_ci_list = []

    #info("Training loop")
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

        info(f"Epoch {epoch} ")
        model.train()
        for X, y, e in train_loader:

            # info(f"Epoch {epoch} - step 1 ")
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            e = e.to(device, dtype=torch.float32)
            # info(f"Epoch {epoch} - step 2 ")
            risk_pred = model(X)
            risk_pred = risk_pred.to(device)
            # info(f"Epoch {epoch} - step 3 ")
            train_loss = criterion(risk_pred, y, e, model)
            # info(f"Epoch {epoch} - step 4 ")
            total_train_loss = total_train_loss + train_loss.item()
            train_c = c_index(-risk_pred, y, e)
            total_train_c = total_train_c + train_c
            # train_ci_list.append(train_c)
            total_train_step += 1.0
            # print ("train_c", train_c)
            # updates parameters
            # info(f"Epoch {epoch} - step 5 ")
            optimizer.zero_grad()
            # info(f"Epoch {epoch} - step 6 ")
            train_loss.backward()
            # info(f"Epoch {epoch} - step 7 ")
            optimizer.step()
            # info(f"Epoch {epoch} - step 8 ")

        train_loss_list.append(total_train_loss/total_train_step)
        train_ci_list.append(total_train_c/total_train_step)

        info("Validation step")
        # validation step
        model.eval()
        with torch.no_grad():
            for X, y, e in val_loader:

                X = X.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                e = e.to(device, dtype=torch.float32)

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

            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            e = e.to(device, dtype=torch.float32)
            risk_pred = model(X)
            risk_pred = risk_pred.to(device)
            test_c = c_index(-risk_pred, y, e)

    test_cm_dict['risk_pred'] = risk_pred.numpy().tolist()
    test_cm_dict['y'] = y.numpy().tolist()
    test_cm_dict['e'] = e.numpy().tolist()
    test_eval_dict['ci'] = test_c.tolist()
    info(f"test ci {test_c} - checkout")

    
    ## Return client's weights (after local training)
    ## See ttps://github.com/itslastonenikhil/federated-learning/blob/main/FederatedLearning.ipynb
    model_params = model.state_dict()
    for entry in model_params:
        model_params[entry] = model_params[entry].cpu().data.numpy().tolist()

    model_params_json = json.dumps(model_params)    
    test_cm_dict = json.dumps(test_cm_dict)
    test_eval_dict = json.dumps(test_eval_dict)

    # Return results to the vantage6 server.
    return {"params": model_params_json,  "client_id":client_id, "num_train_samples": train_data_size, "test_cm": test_cm_dict, "test_eval": test_eval_dict}

