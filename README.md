# FedAvg_vantage6
Implementation of FedAvg on vantage6.

<p align="center">
  <img height="450" src="research_infra_mdt_poc.JPG">
</p>


This repository includes a vantage6-compliant proof-of-concept algorithm for training a (dummy) prediction model using harmonized datasets from the MyDigiTwin project.

### Data pre-processing

This algorithms works with federated datasets (registered within a vantage6 collaboration nodes) that follow the data harmonization approach defined for the MyDigiTwin infrastructure. The algorithm requires, on each node, to pre-process the harmonized datasets with the scripts included on the `preprocessing` folder. This pre-processed CSV file is the one to be enabled on each vantage6 node. This pre-processing is performed as follows:

```bash
python sqlite_to_csv.py <harmonized_db_path> <target_harmonized_csv_file>
python data_prep.py <target_harmonized_csv_file> <target_harmonized_preprocessed_csv_file>
```

For example:
```bash
python sqlite_to_csv.py /data/path/harmonizeddataset.db.sqlite /data/path/harmonizeddataset.db.csv
python data_prep.py /data/path/harmonizeddataset.db.csv /data/path/harmonizeddataset.db.preprocessed.csv
```

### To run the algorithm test:

Install the algorithm as a local package:
`pip install -e .`

Run the test:
`python test/test_lifelines.py --fold_index 0`


# Acknowledgement

_This work (in particular, horizontal splitting on Lifelines for a PoC) is the part of our manuscipt, "MyDigiTwin: A Privacy-Preserving Framework for Personalized Cardiovascular Risk Prediction and Scenario Exploration", that has been submitted to  the special issue on "Building Digital Twins for Personalized Cardiovascular Medicine" in the journal Computers in Biology and Medicine._


_This work is the part of the project MyDigiTwin with project number 628.011.213 of the research programme ``COMMIT2DATA – Big Data & Health'' which is partly financed by the Dutch Research Council (NWO)_

