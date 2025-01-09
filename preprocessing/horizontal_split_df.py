import argparse
import pandas as pd
import os

random_state = 0

def main():

    parser = argparse.ArgumentParser(description="CSV random splitter")
    
    parser.add_argument("input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("output_folder", type=str, help="Output files location")
    
    args = parser.parse_args()

    ## Load CSV file
    df_full = pd.read_csv(args.input_csv)
    print ("df full", df_full)

    ## Add index (pseudo_id)
    # df_full["pseudo_id"] = np.arange(len(df_full))
    print ("df full", df_full)

    ## Horizontal split, n=3 (50%, 30%, 20%)
    df_split_0 = df_full.sample(frac = 0.5, random_state=random_state)

    df_split_half = df_full.drop(df_split_0.index)

    df_split_1 = df_split_half.sample(frac = 0.6, random_state=random_state)
    df_split_2 = df_split_half.drop(df_split_1.index)

    print (df_split_0)
    print (df_split_1)
    print (df_split_2)

    print (len(df_split_0))
    print (len(df_split_1))
    print (len(df_split_2))

    output_file_0 = os.path.join(args.output_folder, f"{args.input_csv}.0.csv")
    output_file_1 = os.path.join(args.output_folder, f"{args.input_csv}.1.csv")
    output_file_2 = os.path.join(args.output_folder, f"{args.input_csv}.2.csv")

    # Save to separate csv files
    df_split_0.to_csv(output_file_0, index=False)
    df_split_1.to_csv(output_file_1, index=False)
    df_split_2.to_csv(output_file_2, index=False)


if __name__ == '__main__':
    main()    

