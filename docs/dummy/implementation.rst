Implementation
==============

Overview
--------
The implementation of our federated learning on Vantage6 comprises the following four python scripts (plus on for performance evaluation):

- central_ci.py: Central part of the algorithm, this script is responsible for the orchestration and aggregation of the algorithm; it also includes the creation of subtasks for all the client nodes.
- partial_risk_prediction.py: Decentral part of the algorithm, this script works for local training followed by returning local weights to the server; it performs the subtasks created in the above cetral script.
- utils.py: Python script for common functions for Partials and performance evaluation.
- networks.py: Python script for creating CVD risk prediction models which is a neural network (based on the python implementation of DeepSurv)




Central (``central``)
-----------------
The central part is responsible for the orchestration and aggregation of the algorithm.

.. Describe the central function here.




Partials
--------
Partials are the computations that are executed on each node. The partials have access
to the data that is stored on the node. The partials are executed in parallel on each
node.

``partial``
~~~~~~~~~~~~~~~~

.. Describe the partial function.

