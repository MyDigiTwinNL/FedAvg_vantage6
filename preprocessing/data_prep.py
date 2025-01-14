import argparse
import numpy as np
import pandas as pd


from datetime import date
from datetime import timedelta
from datetime import datetime


#def replace_missing_values(patients_df):
#    patients_df.replace(to_replace = '\$[0-9]*', value = np.nan, regex = True, inplace = True)
#    return patients_df

def csv_preprocess(input_csv_path:str,output_csv_path:str):

    patient_df = pd.read_csv(input_csv_path)
    print ("patient_df", patient_df)
    print ("len(patient_df)", len(patient_df))

    TEN_YEAR =  timedelta(days = 3650)
    ONE_YEAR = timedelta(days = 365)

    ## One ind who does not have birthdate excluded 
    patient_df = patient_df.dropna(subset=['BIRTHDATE'])


    ## Time origin: date_of_inclusion
    patient_df['BIRTHDATE'] = patient_df['BIRTHDATE'].astype('int').astype('str').astype('datetime64[ns]')
    patient_df['DATE_OF_INCLUSION'] = patient_df['DATE_OF_INCLUSION'].astype('datetime64[ns]')
    patient_df['DATE_OF_LAST_RESPONSE'] = patient_df['DATE_OF_LAST_RESPONSE'].astype('datetime64[ns]')


    patient_df['AGE'] = (patient_df['DATE_OF_INCLUSION']  - patient_df['BIRTHDATE'] ).dt.days / 365

    patient_df['CVD_ONSET_DATE'] = patient_df['CVD_ONSET_DATE'].astype('datetime64[ns]')

    # print ("patient_df['age']", patient_df['age'])

    ### Calculate LENFOL
    ## Exclude patients with prevalent CVD (onset date is earlier than inclusion date)
    patient_df['event_lenfol'] =  (patient_df['CVD_ONSET_DATE']  - patient_df['DATE_OF_INCLUSION']).dt.days 


    patient_df = patient_df.drop(patient_df.index[patient_df['event_lenfol'] < 0])
    print ("len(patient_df)", len(patient_df))

    # print ("num of prevalent cvd: ", all_participant - len(patient_df))
    patient_df['max_lenfol_date'] = patient_df['DATE_OF_INCLUSION'].astype('datetime64[ns]') +pd.to_timedelta(TEN_YEAR, unit = 'D')

    # # calculate LENFOL
    ## Time to event:  date_of_event – date_of_inclusion (in days)
    ## max_follow_up: 10 years (3650 days)
    ## Censoring: min(date_of_death, date_of_last_response, max_follow_up)

    date_cols = ['CVD_ONSET_DATE', 'max_lenfol_date', 'DATE_OF_LAST_RESPONSE']
    patient_df['min_date'] = patient_df[date_cols].min(axis=1)


    patient_df['LENFOL'] = (patient_df['min_date']  - patient_df['DATE_OF_INCLUSION'].astype('datetime64[ns]') ).dt.days


    patient_df['FSTAT'] = patient_df['CVD_STATUS'] 
    patient_df['FSTAT'].replace(to_replace = r'\$[0-9]*', value = np.nan, regex = True, inplace = True)

    patient_df=patient_df.fillna({'FSTAT':0.0})    

    # Change categorical(str) to float
    patient_df['FSTAT'].replace({'Active': 1.0}, inplace=True)   


    ## T2D Status 'Active': 1.0 Nan: 0.0
    patient_df=patient_df.fillna({'T2D_STATUS':0.0})    
    patient_df=patient_df.fillna({'HYPERTENSION_STATUS':0.0})    
    # Change categorical(str) to float
    patient_df['T2D_STATUS'].replace({'Active': 1.0}, inplace=True)   
    patient_df['HYPERTENSION_STATUS'].replace({'Active': 1.0}, inplace=True)   

    ## Smoking, Never smoked tobacco (finding) : 0, Ex-smoker (finding): 1, Smokes tobacco daily (finding): 2
    patient_df['SMOKING_STATUS'].replace({'Never smoked tobacco (finding)': 0.0, 'Ex-smoker (finding)': 1.0, 'Smokes tobacco daily (finding)':2.0}, inplace=True)   

    ## GENDER, Male: 0, Female: 1
    patient_df['GENDER'].replace({'male': 0.0, 'female': 1.0}, inplace=True)   


    # patient_df = patient_df.head(1000)
    # patient_df.to_csv(os.path.join(csv_dir, "patient_t2e_df.csv") )

    predictor_cols = ['GENDER', 'T2D_STATUS', 'SMOKING_STATUS', 'SMOKING_QUANTITY',  
    'TOTAL_CHOLESTEROL_VALUE', 'HDL_CHOLESTEROL_VALUE', 'LDL_CHOLESTEROL_VALUE', 
    'SYSTOLIC_VALUE', 'DIASTOLIC_VALUE', 'PLASMA_ALBUNIM_VALUE', 'EGFR_VALUE', 
    'HEMOGLOBIN_VALUE', 'HYPERTENSION_STATUS', 'HBA1C_VALUE', 'CREATININE_VALUE', 'AGE']

    outcome_cols = ['LENFOL', 'FSTAT']
    print ("len(patient_df)", len(patient_df))
    patient_df = patient_df[predictor_cols+outcome_cols]
    patient_df.to_csv(output_csv_path , index=False)


def main():

    parser = argparse.ArgumentParser(description="CSV data pre-processing")
    
    parser.add_argument("input_csv", type=str, help="Path to input CVS datafile")
    parser.add_argument("output_csv", type=str, help="Path to output (preprocessed) CSV file")
    
    args = parser.parse_args()

    csv_preprocess(args.input_csv,args.output_csv)
    
    

if __name__ == "__main__":
    main()
