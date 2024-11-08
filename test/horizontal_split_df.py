
import os, sys
import pandas as pd
import numpy as np

random_state = 0

def main():

    ## Load CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_filepath = os.path.join(current_dir, "whasncc2.xls")
    df_full = pd.read_excel(data_filepath)
    print ("df full", df_full)

    ## Add index (pseudo_id)
    df_full["pseudo_id"] = np.arange(len(df_full))
    print ("df full", df_full)

    ## Horizontal split, n=3 (50%, 30%, 20%)
    df_split_0 = df_full.sample(frac = 0.5, random_state=random_state)

    df_split_half = df_full.drop(df_split_0.index)

    df_split_1 = df_split_half.sample(frac = 0.6, random_state=random_state)
    df_split_2 = df_split_half.drop(df_split_1.index)

    print (df_split_0)
    print (df_split_1)
    print (df_split_2)

    ## Save to separate csv files
    df_split_0.to_csv("whas_split_0.csv", index=False)
    df_split_1.to_csv("whas_split_1.csv", index=False)
    df_split_2.to_csv("whas_split_2.csv", index=False)




if __name__ == '__main__':
    main()    

