from vantage6.client import UserClient as Client

# Note: we assume here the config.py you just created is in the current directory.
# If it is not, then you need to make sure it can be found on your PYTHONPATH
import config
import configparser


def read_config(ini_file):
    ''' Performs read config file and parses it.

    :param ini_file: (String) the path of a .ini file.
    :return config: (dict) the dictionary of information in ini_file.
    '''
    def _build_dict(items):
        return {item[0]: eval(item[1]) for item in items}
    # create configparser object
    cf = configparser.ConfigParser()
    # read .ini file
    cf.read(ini_file)
    config = {sec: _build_dict(cf.items(sec)) for sec in cf.sections()}
    return config


# Initialize the client object, and run the authentication
client = Client(config.server_url, config.server_port, config.server_api,
                log_level='debug')
client.authenticate(config.username, config.password)

# Optional: setup the encryption, if you have an organization_key
client.setup_encryption(config.organization_key)


predictor_cols = ['GENDER', 'T2D_STATUS', 'SMOKING_STATUS', 'SMOKING_QUANTITY',  
'TOTAL_CHOLESTEROL_VALUE', 'HDL_CHOLESTEROL_VALUE', 'LDL_CHOLESTEROL_VALUE', 
'SYSTOLIC_VALUE', 'DIASTOLIC_VALUE', 'PLASMA_ALBUNIM_VALUE', 'EGFR_VALUE', 
'HEMOGLOBIN_VALUE', 'HYPERTENSION_STATUS', 'CREATININE_VALUE', 'HBA1C_VALUE', 'AGE']

outcome_cols = ['LENFOL', 'FSTAT']

num_update_iter = 21

dl_config = read_config("lifelines_ci.ini")

n_fold = 10
fold_index = 4
output_pth = "aggregated_weights.pth"


model_training_task = client.task.create(
   collaboration=2,
   # Must be set to the 'aggregator' organization
   organizations=[7],
   name="federated_model_training_poc",   
   image="ghcr.io/mydigitwinnl/federated_cvdm_training_poc:c10d8d35725c940101fd7b4d949c5e86e32701bb",
   description='',
   input_= {
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
   databases=[
         {'label': 'lifelines_dummy'}
   ]
)

task_id = model_training_task['id']
print('Waiting for results...')
result = client.wait_for_results(task_id)
print('Results received!')
print(result)


