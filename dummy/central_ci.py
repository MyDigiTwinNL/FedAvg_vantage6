"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""
import os,sys
import json
import torch
import random
import numpy as np
from typing import Any
from .utils import *
from sklearn.metrics import confusion_matrix
from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient

## Set random seed for reproducibility and the same initialization
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
random.seed(0)
np.random.seed(0)


@algorithm_client
def central_ci(
    client: AlgorithmClient, predictor_cols, outcome_cols, dl_config, num_update_iter, n_fold, fold_index, agg_weight_filename
) -> Any:
    '''
    AlgorithmClient:
    predictor_cols: a python list of selected predictors after further data preparation step by "data_prep.py" from the harmonized FHIR data 
    outcome_cols: a python list of outcome variables after further data preparation step by "data_prep.py" from the harmonized FHIR data, it should contrain 'LENFOL'(length of follow-up) and 'FSTAT' (event occurence)
    dl_config: hyperparameter values regarding a neural network architecture and its training (specified in the configuration ini file)
    num_update_iter: the number of aggregation iterations (set to 20 in our PoC)
    n_fold: the number of folds in the data splitting in each client for train/valid/test(set to 10 in our PoC)
    (optional) fold_index: the index of the fold that will be used for test (it is for the corrected resampled t-test in the performance evaluation)
    agg_weight_filename: Path to output pth datafile which contains the aggregated wegiths after the full iterations of FL

    '''

    """ Central part of the algorithm """
    # central function.
    # get all organizations (ids) within the collaboration so you can send a task to them.
    organizations = client.organization.list()
    org_ids = [organization.get("id") for organization in organizations]

    global_ci_list = [] # List of C-statistic for performance evaluation
    local_ci_list = [] # List of C-statistic for performance evaluation

    ## Iterations of broadcasting weights/local weight updating/sending back/aggregation
    for i in range(num_update_iter):
        '''
        "avged_params" is an aggregated weight
        '''
        if i == 0:
            # at the first iteration, the aggregated weight should be None
            avged_params = None 
        else:
            # from the second iteration, it contained the weighted average of the model weight, it should be json serialized to be broadcasted to the clients again
            avged_params = dict2json(avged_params) 

        # Define input parameters for a subtask
        info("Defining input parameters")
        input_ = {
            "method": "partial_risk_prediction",
            "kwargs": {
                "predictor_cols": predictor_cols,
                "outcome_cols": outcome_cols,
                "dl_config": dl_config,
                "avged_params": avged_params,
                "update_iter": i,
                "n_fold": n_fold,
                "fold_index": fold_index
            }
        }

        # Create a subtask for all organizations in the collaboration.
        info("Creating subtask for all organizations in the collaboration")
        task = client.task.create(
            input_=input_,
            organizations=org_ids,
            name="FedAvg_MDT",
            description="Training task on each client"
        )

        # wait for node to return results of the subtask.
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        params_list =[] # params here indicate network parameters; here, model weight
        num_train_samples_list = [] # the number of training samples for weighted averaging
        test_risk_pred_list = [] # risk prediction results on test data 
        test_y_list = [] # ground truth of followup length (label) of test data (only for evaluation)
        test_e_list = [] # ground truth of event occurence (label) of test data (only for evaluation)
        test_ci_list = [] # C-statistic based on local model performance evaluation results

        # process the serialized json data from client (returned after completing the subtask by client based on the partial function)
        for r in results:
            params = r["params"]
            num_train_samples = r["num_train_samples"]
            params = json.loads(params) 

            test_cm = r["test_cm"]
            # test_cm = json2dict(test_cm)
            test_cm = json.loads(test_cm) 

            test_eval = r["test_eval"]
            # test_eval = json2dict(test_eval)
            test_eval = json.loads(test_eval) 

            for entry in params:
                # params[entry] = torch.from_numpy(np.array(params[entry]))
                params[entry] = np.array(params[entry])


            test_risk_pred_list.append(test_cm['risk_pred'])
            test_y_list.append(test_cm['y'])
            test_e_list.append(test_cm['e']) 
            test_ci_list.append(test_eval['ci'])
            params_list.append(params)
            num_train_samples_list.append(num_train_samples)

        ## Weighed averging of the weight from local models
        # See fed_avg function in utils.py
        avged_params = fed_avg(params_list, num_train_samples_list)
        
        ## All the lines below are merely for performance evaluation
        ## Collect all the prediction results 
        risk_pred_stack = np.concatenate( test_risk_pred_list, axis=0 )

        y_stack = np.concatenate( test_y_list, axis=0 )
        e_stack = np.concatenate( test_e_list, axis=0 )

        glo_ci = c_index(-risk_pred_stack, y_stack, e_stack)
        global_ci_list.append(glo_ci)
        local_ci_list.append(test_ci_list)

        ## Plot training curves
        current_dir = os.path.dirname(os.path.abspath(__file__))
        figure_result_dir = os.path.join(current_dir, "figure_ci_results_%s" %fold_index)
        if not os.path.exists(figure_result_dir):
            os.makedirs(figure_result_dir)

        plot_global_results(global_ci_list, figure_result_dir, "C-statistic")
        plot_local_results(local_ci_list, figure_result_dir, "C-statistic")



    # Save the aggregated weights of the network after the max update iterations of FL
    torch.save(avged_params, agg_weight_filename)

    # Create a folder for saving results for the corrected resampled t-test
    ttest_dir = os.path.join(current_dir, "ttest_ci")
    if not os.path.exists(ttest_dir):
        os.makedirs(ttest_dir)

    # Save C-statistic results for for the corrected resampled t-test
    np.save(os.path.join(ttest_dir, "global_ci_%s.npy" %fold_index ), np.array(global_ci_list))
    np.save(os.path.join(ttest_dir, "local_ci_%s.npy" %fold_index), np.array(local_ci_list))

    return i

