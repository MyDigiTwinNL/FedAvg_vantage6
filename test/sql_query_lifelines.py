import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

### DB file path ###
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(current_dir, "csv")
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
project_dir = os.path.dirname(os.path.abspath(__file__ + "/../../../"))
db_dir = os.path.join(project_dir, "pheno_lifelines_sqlite")
db_filepath = os.path.join(db_dir, "db-lifelines.db")
print ("db_filepath", db_filepath)
# Connect to the SQLite database
conn = sqlite3.connect(db_filepath)  


# ## Dump the table 'PATIENT' into a dataframe
# sql_query = "SELECT * FROM PATIENTS P;"
# sql_col = "SELECT c.name FROM pragma_table_info('PATIENTS') c;"
# cursor = conn.execute(sql_query)

# patient_df = pd.DataFrame(cursor.fetchall())
# patient_df.rename(columns={"ID": "PATIENTID"})

# print (patient_df)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# print(cursor.fetchall())
tables = cursor.fetchall()

for t in tables:
    table_name = t[0]
    print ("table_name", table_name)
    if table_name == "PATIENTS":
        patient_df = pd.read_sql("SELECT * from %s" %table_name, conn)
        patient_df.rename(columns={"ID": "PATIENTID"}, inplace=True)
        patient_df = patient_df.drop_duplicates(subset='PATIENTID', keep="last")
        print (patient_df)
        print ("len(patient_df)", len(patient_df))
        print (patient_df.columns)
        continue

    if table_name == "sqlite_stat1":
        continue
    

    variable_df = pd.read_sql("SELECT * from %s" %table_name, conn) # ['VALUE', 'EFFECTIVE_DATE', 'UNIT', 'PATIENTID']
    print (variable_df)
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
    print (variable_df)
    print ("len(variable_df)", len(variable_df))
    ## merge patient_df with variable_df based on 'PATIENTID'
    patient_df = pd.merge(patient_df, variable_df, on = 'PATIENTID', how = "left", suffixes = ('', f'_{table_name}')) 
    print ("patient_df", patient_df)
    print ("len(patient_df)", len(patient_df))
    print ("patient_df.columns", patient_df.columns)
    
print ("len(patient_df)", len(patient_df))
patient_df.to_csv(os.path.join(csv_dir, "patient_df.csv") )
patient_exmaple_df = patient_df.head(100)
patient_exmaple_df.to_csv(os.path.join(csv_dir, "patient_example_df.csv") )

# ## Dump the table 'PATIENT' into a dataframe
# sql_query = "SELECT * FROM PATIENTS P;"
# sql_col = "SELECT c.name FROM pragma_table_info('PATIENTS') c;"
# cursor = conn.execute(sql_query)

# patient_df = pd.DataFrame(cursor.fetchall())
# print (patient_df)
# cursor = conn.execute(sql_col)
# table_cols = [row[0] for row in cursor.fetchall()]

# patient_df.columns = table_cols
# print (patient_df)
# print ("patient_df.columns", patient_df.columns)
# patient_df.to_csv('db_sql_lifelines.csv')









## Dump the table 'PATIENT' into a dataframe


##CVD ( a composite of MI, HF, and Stroke)



# # # Define the SQL query to retrieve birth years of patients with T2D
# # sql_query = "SELECT BIRTHDATE FROM PATIENTS WHERE T2D_STATUS = 'Active';"

# # Select all the columns of all the patients
# sql_query = "SELECT * FROM PATIENTS P;"

# sql_col = "SELECT c.name FROM pragma_table_info('PATIENTS') c;"

# sql_col = "SELECT c.name FROM pragma_table_info('SMOKING') c;"

# # Execute the query and fetch the birth years into a list
# cursor = conn.execute(sql_col)

# # for row in cursor.fetchall():
# #     print ("row", row)



# # birth_years = [int(row[0]) for row in cursor.fetchall()]
# # birth_years = [row[0] for row in cursor.fetchall()]
# # print ("birth_years", birth_years)


# table_to_list = [row[0] for row in cursor.fetchall()]
# print (table_to_list)
# # Close the database connection
# conn.close()

# # Calculate ages from birth years and current year
# current_year = datetime.now().year
# ages = [current_year - birth_year for birth_year in birth_years]

# # Calculate median and standard deviation using NumPy
# median_age = np.median(ages)
# std_dev_age = np.std(ages)

# # Display the results
# print(f"Median Age of Patients with T2 Diabetes: {median_age} years")
# print(f"Standard Deviation of Age: {std_dev_age:.2f} years")
