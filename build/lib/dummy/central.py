"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""
from typing import Any

from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient


@algorithm_client
def central(
    client: AlgorithmClient, column_name
) -> Any:

    """ Central part of the algorithm """
    # TODO implement this function. Below is an example of a simple but typical
    # central function.

    # get all organizations (ids) within the collaboration so you can send a
    # task to them.
    organizations = client.organization.list()
    org_ids = [organization.get("id") for organization in organizations]

    # Define input parameters for a subtask
    info("Defining input parameters")
    input_ = {
        "method": "partial",
        "kwargs": {
            # TODO add sensible values
            "column_name": column_name,

        }
    }

    # create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=org_ids,
        name="My subtask",
        description="This is a very important subtask"
    )


    # wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    total_sum = 0
    total_count = 0 

    for r in resukts:
        total_sim += r["sum"]
        total_count += r["count"]

    #  return {}
    return total_sum / total_count

    # TODO probably you want to aggregate or combine these results here.
    # For instance:
    # results = [sum(result) for result in results]

    # return the final results of the algorithm
    return results

# TODO Feel free to add more central functions here.
