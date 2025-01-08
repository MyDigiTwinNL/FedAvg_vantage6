import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from pandas import DataFrame


def sql_to_dataframe(db_file_path:str)->DataFrame:

    conn = sqlite3.connect(db_file_path)  
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # print(cursor.fetchall())
    tables = cursor.fetchall()

    for t in tables:
        table_name = t[0]
        #print ("table_name", table_name)
        if table_name == "PATIENTS":
            patient_df = pd.read_sql("SELECT * from %s" %table_name, conn)
            patient_df.rename(columns={"ID": "PATIENTID"}, inplace=True)
            patient_df = patient_df.drop_duplicates(subset='PATIENTID', keep="last")
            #print (patient_df)
            #print ("len(patient_df)", len(patient_df))
            #print (patient_df.columns)
            continue

        if table_name == "sqlite_stat1":
            continue
        
        variable_df = pd.read_sql("SELECT * from %s" %table_name, conn) # ['VALUE', 'EFFECTIVE_DATE', 'UNIT', 'PATIENTID']
        #print (variable_df)
        # print (variable_df.columns)
        if table_name == "SMOKING":        
            variable_df = variable_df[['STATUS' , 'QUANTITY', 'PATIENTID']]
            variable_df.rename(columns={"STATUS": '%s_STATUS' %table_name, "QUANTITY": '%s_QUANTITY' %table_name}, inplace=True)
            

        elif table_name == "BLOODPRESSURE":
            variable_df = variable_df[['SYSTOLIC_VALUE', 'DIASTOLIC_VALUE', 'PATIENTID']]
            # variable_df.rename(columns={"SYSTOLIC": 'SYSTOLIC_VALUE', "DIASTOLIC": 'DIASTOLIC_VALUE'}, inplace=True)
        else:
            variable_df = variable_df[['VALUE', 'PATIENTID']]
            variable_df.rename(columns={"VALUE": '%s_VALUE' %table_name}, inplace=True)
        
        # print (variable_df.columns)

        variable_df = variable_df.drop_duplicates(subset='PATIENTID', keep="last")
        #print (variable_df)
        #print ("len(variable_df)", len(variable_df))
        ## merge patient_df with variable_df based on 'PATIENTID'
        patient_df = pd.merge(patient_df, variable_df, on = 'PATIENTID', how = "left", suffixes = ('', f'_{table_name}')) 

        return patient_df


def main():

    parser = argparse.ArgumentParser(description="SQLite query to CSV")
    
    parser.add_argument("input_db", type=str, help="Path to input SQLite database")
    parser.add_argument("output_csv", type=str, help="Path to output CSV file")
    
    args = parser.parse_args()

    patient_df = sql_to_dataframe(args.input_db)

    print ("len(patient_df)", len(patient_df))

    patient_df.to_csv(args.output_csv)

    

if __name__ == "__main__":
    main()
