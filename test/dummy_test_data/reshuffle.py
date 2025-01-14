# python program that reads a CSV file and that updates the cells as follows:

# - If the cell contains a data of the form YYYY-MM it replaces it with a random date between 1990-1 and 2000-1
# - if the column name ends with is 'BIRTHDATE', replace the value with a random integer between 1910 and 1933
# - if the column name ends with is 'DECEASED_DATE', replace the value with a random date between 2001-1 and 2020-1
# - if the column name of the cell ends with '_ONSET_DATE', replace the value with a random integer between 1960 and 2000

# - If the row is on the columns TOTAL_CHOLESTEROL_VALUE, HDL_CHOLESTEROL_VALUE, LDL_CHOLESTEROL_VALUE, SYSTOLIC_VALUE, DIASTOLIC_VALUE, PLASMA_ALBUMIN,  EGFR_VALUE, HEMOGLOBIN_VALUE, CREATININE_VALUE or HBA1C_VALUE, generate a random value that is medically sound for such variables.
# - If the row is on the column HDL_CHOLESTEROL_VALUE generate a random value that is consistent for total cholesterol


import pandas as pd
import random
from datetime import datetime, timedelta

def random_date(start_year, end_year):
    """Generate a random date between start_year-1-1 and end_year-1-1"""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 1, 1)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days - 1)
    return (start_date + timedelta(days=random_days)).strftime('%Y-%m')

def random_date_between(start_date, end_date):
    """Generate a random date between two given dates"""
    start_date = datetime.strptime(start_date, '%Y-%m')
    end_date = datetime.strptime(end_date, '%Y-%m')
    delta = end_date - start_date
    random_days = random.randint(0, delta.days - 1)
    return (start_date + timedelta(days=random_days)).strftime('%Y-%m')

def update_csv(file_path, output_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # List of medically sound ranges for various variables
    medical_ranges = {
        "TOTAL_CHOLESTEROL_VALUE": (150, 250),
        "HDL_CHOLESTEROL_VALUE": (40, 90),
        "LDL_CHOLESTEROL_VALUE": (50, 150),
        "SYSTOLIC_VALUE": (90, 140),
        "DIASTOLIC_VALUE": (60, 90),
        "PLASMA_ALBUMIN": (3.5, 5.5),
        "EGFR_VALUE": (60, 120),
        "HEMOGLOBIN_VALUE": (12, 17),
        "CREATININE_VALUE": (0.5, 1.5),
        "HBA1C_VALUE": (4, 6),
    }

    # Iterate through columns and rows
    for column in df.columns:
        if column.endswith("BIRTHDATE"):
            # Random birth year between 1910 and 1933
            df[column] = df[column].apply(lambda _: random.randint(1910, 1933))

        elif column.endswith("DECEASED_DATE"):
            # Random date between 2001 and 2020
            df[column] = df[column].apply(lambda _: random_date(2001, 2020))

        elif column.endswith("_ONSET_DATE"):
            # Random year between 1960 and 2000
            df[column] = df[column].apply(lambda _: random.randint(1960, 2000))

        elif column in medical_ranges:
            # Generate a medically sound value
            low, high = medical_ranges[column]
            df[column] = df[column].apply(lambda _: round(random.uniform(low, high), 2))

            # Special case for HDL_CHOLESTEROL_VALUE to maintain consistency
            if column == "HDL_CHOLESTEROL_VALUE":
                total_cholesterol = df["TOTAL_CHOLESTEROL_VALUE"]
                df[column] = df[column].apply(lambda hdl: min(hdl, total_cholesterol.mean() / 3))

        elif df[column].astype(str).str.match(r"^\\d{4}-\\d{2}$").any():
            # Replace YYYY-MM with random date between 1990-1 and 2000-1
            df[column] = df[column].apply(
                lambda value: random_date(1990, 2000) if pd.notna(value) and isinstance(value, str) and value.match(r"^\\d{4}-\\d{2}$") else value
            )

    # Handle specific column logic
    if "DATE_OF_INCLUSION" in df.columns and "BIRTHDATE" in df.columns:
        df["DATE_OF_INCLUSION"] = df.apply(
            lambda row: f"{int(row['BIRTHDATE']) + random.randint(20, 30)}-01" if not pd.isna(row["BIRTHDATE"]) else None,
            axis=1
        )

    if "CVD_ONSET_DATE" in df.columns and "DATE_OF_INCLUSION" in df.columns and "DECEASED_DATE" in df.columns:
        df["CVD_ONSET_DATE"] = df.apply(
            lambda row: random_date_between(row["DATE_OF_INCLUSION"], row["DECEASED_DATE"]) if not pd.isna(row["DATE_OF_INCLUSION"]) and not pd.isna(row["DECEASED_DATE"]) else None,
            axis=1
        )

    # Write the updated DataFrame to a new CSV file
    df.to_csv(output_path, index=False)


# Example usage
input_file = "fhir.dummydata.10k.csv"  # Replace with the path to your CSV file
output_file = "fhir.dummydata.10k.v2.csv"  # Replace with the desired output path
update_csv(input_file, output_file)

print(f"Updated CSV file has been saved to {output_file}")

