"""
Run this script to test your algorithm locally (without building a Docker
image) using the mock client.

Run as:

    python test.py

Make sure to do so in an environment where `vantage6-algorithm-tools` is
installed. This can be done by running:

    pip install vantage6-algorithm-tools
"""
import os,sys

sys.path.append('../')

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print (sys.path)

import pandas as pd
from dummy.utils import read_config
import argparse
# get path of current directory



current_path = Path(__file__).parent
current_dir = os.path.dirname(os.path.abspath(__file__))
config_ini_filepath = os.path.join(current_dir, "lifelines_ci.ini")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n_fold', default= 10, help='k of k-fold for ci', type=int)
    parser.add_argument('--fold_index', help="fold index out of k", type=int, required = True)
    parser.add_argument("--output_pth", default= "aggregated_weights.pth", type=str, help="Path to output pth datafile which contains the aggregated wegiths after the full iterations of FL")

    # return parser.parse_args()

    args = parser.parse_args()
    # print("Arguments:",args)
    n_fold = args.n_fold
    fold_index = args.fold_index
    output_pth = args.output_pth

    ## Mock client
    ## Horizontal splitted data, n=3 (50%, 30%, 20%)
    client = MockAlgorithmClient(
        datasets=[
            # Data for first organization
            [{
                "database": str(current_path/"dummy_test_data"/"fhir.dummydata.10k.preprocessed.csv.0.csv"),
                "db_type": "csv",
                "input_data": {}
            }],
            # Data for second organization
            [{
                "database": str(current_path/"dummy_test_data"/"fhir.dummydata.10k.preprocessed.csv.1.csv"),
                "db_type": "csv",
                "input_data": {}
            }],
            # Data for third organization
            [{
                "database": str(current_path/"dummy_test_data"/"fhir.dummydata.10k.preprocessed.csv.2.csv"),
                "db_type": "csv",
                "input_data": {}
            }],
        ],
        module="dummy"
    )




    ## get column name
    ## Description regarding benchmark, https://web.archive.org/web/20170515104524/http://www.umass.edu/statdata/statdata/data/whasncc2.txt
    ##df_template = pd.read_csv(str(current_path/"dummy_test_data"/"fhir.dummydata.10k.preprocessed.csv.0.csv"))
    ##list_of_column_names = list(df_template.columns)
    # print ("list_of_column_names", list_of_column_names)

    predictor_cols = ['GENDER', 'T2D_STATUS', 'SMOKING_STATUS', 'SMOKING_QUANTITY',  
'TOTAL_CHOLESTEROL_VALUE', 'HDL_CHOLESTEROL_VALUE', 'LDL_CHOLESTEROL_VALUE', 
'SYSTOLIC_VALUE', 'DIASTOLIC_VALUE', 'PLASMA_ALBUNIM_VALUE', 'EGFR_VALUE', 
'HEMOGLOBIN_VALUE', 'HYPERTENSION_STATUS', 'CREATININE_VALUE', 'HBA1C_VALUE', 'AGE']
    outcome_cols = ['LENFOL', 'FSTAT']

    num_update_iter = 21
    # num_update_iter = 4

    # list mock organizations
    organizations = client.organization.list()
    org_ids = [organization["id"] for organization in organizations]
    print ("org_ids", org_ids)

    # Configuration regarding deep learning (neural network and its training)
    dl_config = read_config(config_ini_filepath)

    # Run the central method on 1 node and get the results
    central_task = client.task.create(
        input_={
            "method":"central_ci",
            "kwargs": {
                "predictor_cols": predictor_cols,
                "outcome_cols": outcome_cols,
                "dl_config": dl_config,
                "num_update_iter": num_update_iter,
                "n_fold": n_fold,
                "fold_index": fold_index,
                "agg_weight_filename": output_pth
            }
        },
        # organizations=[org_ids[0]],
        organizations=[1],
    )

    results = client.wait_for_results(central_task.get("id"))
    print(results)





if __name__ == '__main__':
    main()    

