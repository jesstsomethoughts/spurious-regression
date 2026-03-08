# The goal of this script is to split the Code Metrics dataset (all split='train') into a train and test sets. The training set should follow the natural distribution of the data, while the test set should be generally uniform across all attributes. 

import argparse
from ast import literal_eval
from os.path import join
import random
import json
import numpy as np
import pandas as pd
from datasets import load_dataset # HuggingFace datasets library
from sklearn.model_selection import train_test_split

# Add args so we can potentially apply this to other datasets
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Specify dataset
parser.add_argument('--dataset', type=str, default='akhauriyash/Code-Regression', help="Path to HuggingFace dataset")
parser.add_argument('--cols', nargs='+', default=['target', 'space', 'metadata'], help="List of columns to take from dataset")
parser.add_argument('--space', type=str, default='CDSS', help="Space to split/filter dataset on")
parser.add_argument('--target_col_name', type=str, default='target', help="Name of the target metric column")
parser.add_argument('--test_size', type=int, default=100, help='Number of datapoints in each test set split')
parser.add_argument('--test_split', type=int, default=20, help="Percent of data to designate as test set")
parser.add_argument('--attr', type=str, default='language', help="Attribute by which we want to stratify dataset on")
parser.add_argument('--save', type=bool, default=True, help="Save DFs generated to a CSV file")
parser.add_argument('--data_path', type=str, default='data', help="Path to save CSV files to")
parser.add_argument('--filter_low_attrs', type=bool, default=True, help="Filter out attributes with low # of total datapoints")
parser.add_argument('--verbose', type=bool, default=True, help="Print out extra logging statements")

# Init args
parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()

'''
Filters the dataset by the top 1% of target metric values and processes the metadata string fields to create new columns per field
args:
df: the dataframe to process
'''
def filter_dataset(df):
    space_df = df[df['space'] == args.space].copy()
    # Filter out top 1% of data based on "target" metric
    q_99 = space_df[f'{args.target_col_name}'].quantile(0.99)
    filtered_df = space_df[space_df[f'{args.target_col_name}'] <= q_99].copy()

    # Convert string to dicts for the "metadata" field to separate out each property
    if 'metadata' in args.cols:
        # Define dict for all fields in metadata string
        md_dict = filtered_df['metadata'].map(lambda x: json.loads(x.replace("'", '"')))
        
        # Build individual cols for each field in metadata
        metadata_df = pd.DataFrame(md_dict.tolist(), index=filtered_df.index)
        # Drop extra metadata col and join with 
        filtered_df = filtered_df.drop('metadata', axis=1).join(metadata_df)
    
    return filtered_df

'''
Creates a balanced test set from the given dataframe. All other datapoints go into the training set.
args:
df: the dataframe to process
max_size: the max size of the test set for each attribute value
seed: random seed value to start from
verbose: to print out extra debug lines or not
'''
def make_balanced_testset(df, max_size=args.test_size, seed=700, verbose=args.verbose):
    test_set = [] # Init test set
    random.seed(seed) # Generate random number

    # If an attribute doesn't have enough datapoints, filter it from the dataset
    if args.filter_low_attrs:
        counts = df[args.attr].value_counts()
        keep_attrs = counts[counts >= args.test_size].index
        df = df[df[args.attr].isin(keep_attrs)]
    
    # Split target metric of original dataset into bins
    df['target_bin'] = (np.log10(df['target'] + 1) * 100).round().astype(int)

    # Get list of unique attributes
    attributes = df[f"{args.attr}"].unique()
    num_attrs = len(attributes)
    print(f"Total # of unique attributes {args.attr}: {num_attrs}")

    # Iterate through all attributes to add to test set
    for attr in attributes:
        attr_df = df[df[args.attr] == attr] # Save DF of only rows with that attr
        # Iterate through target bins for each attribute
        relevant_bins = attr_df['target_bin'].unique() # only take target bins that exist
        # Calculate samples per bin per attr df
        # samples_per_bin = max(1, max_size // len(relevant_bins))
        total_allowed_test = min(len(attr_df) * (args.test_split / 100), args.test_size)
        samples_per_bin = max(1, int(total_allowed_test // len(relevant_bins)))
        for bin_val in relevant_bins:
            # Filter out new "curr_df" by target bin values so we can take a certain # from each bin to keep the test set balanced
            curr_df = attr_df[attr_df['target_bin'] == bin_val]
            # Index curr_df
            curr_data = curr_df.index.tolist()
            random.shuffle(curr_data)
            # Calculate split size – preserve % of test split
            max_allowed_from_bin = int(len(curr_data) * (args.test_split / 100))
            curr_size = min(samples_per_bin, max_allowed_from_bin)
            if len(curr_data) > 0 and curr_size == 0 and samples_per_bin > 0:
                curr_size = 1
            if len(curr_data) < curr_size:
                print(f"len curr data {len(curr_data)}")
                print(f"samples per bin: {curr_size}")
            test_set += list(curr_data[:curr_size])
    
    if verbose:
        print(f"# of Datapoints in Test Set: {len(test_set)}")

    # Update dataframe with "split" col to denote which rows are "train" and which are "test"
    df['split'] = 'train' # Default to train
    df.loc[test_set, 'split'] = 'test' 

    if verbose:
        print(df.head())

    # to do: also create val test??
    return df

def main():
    # Load dataset, select columns, and convert to a Pandas DataFrame
    df = load_dataset(args.dataset, split="train").to_pandas()[args.cols]
    if args.verbose:
        print("Dataset loaded!")
        print(df.head())

    # Filter dataset by space
    filtered_df = filter_dataset(df)
    if args.verbose:
        print("Dataset filtered and processed!")
        print(filtered_df.head())

    # Split dataset into train and test, with a uniform distribution of target metrics across attributes for test dataset
    split_df = make_balanced_testset(filtered_df)
    if args.verbose:
        print("Dataset split!")

    # Save datasets to CSV files if args.save is True
    if args.save:
        split_df.to_csv(str(join(args.data_path, f"split_df.csv")), index=False)
    
    return split_df


if __name__ == '__main__':
    main()

