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
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from pathlib import Path
import pandas as pd
from dummy.utils import read_config

# get path of current directory
current_path = Path(__file__).parent
current_dir = os.path.dirname(os.path.abspath(__file__))
config_ini_filepath = os.path.join(current_dir, "whas.ini")

def main():

    ## Mock client
    ## Horizontal splitted data, n=3 (50%, 30%, 20%)
    client = MockAlgorithmClient(
        datasets=[
            # Data for first organization
            [{
                "database": current_path / "whas_split_0.csv",
                "db_type": "csv",
                "input_data": {}
            }],
            # Data for second organization
            [{
                "database": current_path / "whas_split_1.csv",
                "db_type": "csv",
                "input_data": {}
            }],
            # Data for third organization
            [{
                "database": current_path / "whas_split_2.csv",
                "db_type": "csv",
                "input_data": {}
            }],
        ],
        module="dummy"
    )

    ## get column name
    ## Description regarding benchmark, https://web.archive.org/web/20170515104524/http://www.umass.edu/statdata/statdata/data/whasncc2.txt
    df_template = pd.read_csv('whas_split_0.csv')
    list_of_column_names = list(df_template.columns)
    # print ("list_of_column_names", list_of_column_names)

    predictor_cols = ['AGE', 'SEX', 'BMI', 'CHF', 'MIORD']
    outcome_cols = ['LENFOL', 'FSTAT']

    num_update_iter = 20

    # list mock organizations
    organizations = client.organization.list()
    org_ids = [organization["id"] for organization in organizations]
    print ("org_ids", org_ids)

    # Configuration regarding deep learning (neural network and its training)
    dl_config = read_config(config_ini_filepath)

    # Run the central method on 1 node and get the results
    central_task = client.task.create(
        input_={
            "method":"central",
            "kwargs": {
                "predictor_cols": predictor_cols,
                "outcome_cols": outcome_cols,
                "dl_config": dl_config,
                "num_update_iter": num_update_iter
            }
        },
        # organizations=[org_ids[0]],
        organizations=[1],
    )

    results = client.wait_for_results(central_task.get("id"))
    print(results)

    # # Run the partial method for all organizations
    # task = client.task.create(
    #     input_={
    #         "method":"partial",
    #         "kwargs": {
    #             # TODO add sensible values
    #             "column_name": "Age",

    #         }
    #     },
    #     organizations=org_ids
    # )
    # print(task)

    # # Get the results from the task
    # results = client.wait_for_results(task.get("id"))
    # print(results)


if __name__ == '__main__':
    main()    

