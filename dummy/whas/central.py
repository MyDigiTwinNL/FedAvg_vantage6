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
from sklearn.metrics import accuracy_score, roc_curve, auc, recall_score
from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient


torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
random.seed(0)
np.random.seed(0)


@algorithm_client
def central(
    client: AlgorithmClient, predictor_cols, outcome_cols, dl_config, num_update_iter
) -> Any:

    """ Central part of the algorithm """
    # TODO implement this function. Below is an example of a simple but typical
    # central function.

    # get all organizations (ids) within the collaboration so you can send a
    # task to them.
    organizations = client.organization.list()
    org_ids = [organization.get("id") for organization in organizations]

    global_acc_list = []
    global_spe_list = []
    global_sen_list = []
    global_auc_list = []

    local_acc_list = []
    local_spe_list = []
    local_sen_list = []
    local_auc_list = []


    for i in range(num_update_iter):
        print ("update iteration: ", i)

        if i == 0:
            avged_params = None
        else:
            avged_params = dict2json(avged_params)

        # Define input parameters for a subtask
        info("Defining input parameters")
        input_ = {
            "method": "partial",
            "kwargs": {
                "predictor_cols": predictor_cols,
                "outcome_cols": outcome_cols,
                "dl_config": dl_config,
                "avged_params": avged_params,
                "update_iter": i
            }
        }


        # create a subtask for all organizations in the collaboration.
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


        params_list =[]
        num_train_samples_list = []

        test_tn_list = []
        test_fp_list = []
        test_fn_list = []
        test_tp_list = []
        test_acc_list = []
        test_spe_list = []
        test_sen_list = []
        test_auc_list = []

        test_y_list = []
        test_pred_list = []


        for r in results:
            params = r["params"]
            num_train_samples = r["num_train_samples"]
            params = json.loads(params) 

            test_cm = r["test_cm"]
            test_cm = json.loads(test_cm) 
            test_eval = r["test_eval"]
            test_eval = json.loads(test_eval) 

            test_pred = r["test_pred"] 
            test_pred = json.loads(test_pred) 


            for entry in params:
                # params[entry] = torch.from_numpy(np.array(params[entry]))
                params[entry] = np.array(params[entry])


            test_tn_list.append(test_cm['tn'])
            test_fp_list.append(test_cm['fp'])
            test_fn_list.append(test_cm['fn'])
            test_tp_list.append(test_cm['tp'])
            test_acc_list.append(test_eval['accuracy'])
            test_spe_list.append(test_eval['specificity'])
            test_sen_list.append(test_eval['sensitivity'])
            test_auc_list.append(test_eval['auc_value'])

            test_y_list.append(test_pred['y_test'])
            test_pred_list.append(test_pred['prediction_results'])


            params_list.append(params)
            num_train_samples_list.append(num_train_samples)

        avged_params = fed_avg(params_list, num_train_samples_list)
        # print ("avged_params", avged_params)
        

        ## Compute global results
        tn = sum(test_tn_list)
        fp = sum(test_fp_list)
        fn = sum(test_fn_list)
        tp = sum(test_tp_list)

        y_stack = np.concatenate( test_y_list, axis=0 )
        pred_stack = np.concatenate( test_pred_list, axis=0 )

        fpr, tpr, thresholds = roc_curve(y_stack, pred_stack)
        glo_auc = auc(fpr, tpr)

        glo_accuracy = (tp+tn) / (tp+fp+fn+tn)
        glo_specificity = tn / (tn + fp)
        glo_sensitivity = tp / (tp + fn) 

        global_acc_list.append(glo_accuracy)
        global_spe_list.append(glo_specificity)
        global_sen_list.append(glo_sensitivity)
        global_auc_list.append(glo_auc)

        local_acc_list.append(test_acc_list)
        local_spe_list.append(test_spe_list)
        local_sen_list.append(test_sen_list)
        local_auc_list.append(test_auc_list)

        ## Plot training curves
        current_dir = os.path.dirname(os.path.abspath(__file__))
        figure_result_dir = os.path.join(current_dir, "figure_results_")
        if not os.path.exists(figure_result_dir):
            os.makedirs(figure_result_dir)

        plot_global_results(global_acc_list, figure_result_dir, "accuracy")
        plot_global_results(global_spe_list, figure_result_dir, "specificity")
        plot_global_results(global_sen_list, figure_result_dir, "sensitivity")
        plot_global_results(global_auc_list, figure_result_dir, "AUC")

        plot_local_results(local_acc_list, figure_result_dir, "accuracy")
        plot_local_results(local_spe_list, figure_result_dir, "specificity")
        plot_local_results(local_sen_list, figure_result_dir, "sensitivity")
        plot_local_results(local_auc_list, figure_result_dir, "AUC")

            # model_state_dict = json.loads(r)
            # print ("model_state_dict", model_state_dict)
            # total_sum += r["sum"]
            # total_count += r["count"]

        #  return {}

        print ("global_acc_list", global_acc_list)
        print ("local_acc_list", local_acc_list)

        print ("global_auc_list", global_auc_list)
        print ("local_auc_list", local_auc_list)


    return i




    # TODO probably you want to aggregate or combine these results here.
    # For instance:
    # results = [sum(result) for result in results]

    # return the final results of the algorithm


# TODO Feel free to add more central functions here.
