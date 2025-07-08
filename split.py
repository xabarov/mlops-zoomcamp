"""
Split the data into training, validation and test sets
"""
import os

import numpy as np
import pandas as pd


def split(val_portion: float = 0.1, test_portion: float = 0.1, save_path: str = 'nyc/split'):
    """
    Split the data into training, validation and test sets
    """

    df = pd.read_parquet('nyc/processed/yellow_tripdata_processed.parquet')

    if not os.path.exists("nyc/split"):
        os.makedirs("nyc/split")

    df_train, df_val_and_test = np.split(df.sample(frac=1, random_state=42), [
                                         int((1 - val_portion - test_portion) * len(df))])
    df_val, df_test = np.split(df_val_and_test.sample(frac=1, random_state=42), [
                               int((1 - test_portion) * len(df_val_and_test))])

    # save in split folder all df
    df_train.to_parquet(f'{save_path}/train.parquet')
    df_val.to_parquet(f'{save_path}/val.parquet')
    df_test.to_parquet(f'{save_path}/test.parquet')


if __name__ == "__main__":

    # argparse for val_portion and test_portion
    import argparse

    parser = argparse.ArgumentParser(
        description='Split the dataset into train, validation and test sets.')
    parser.add_argument('--val_portion', type=float, default=0.1,
                        help='Portion of the data to use for validation')
    parser.add_argument('--test_portion', type=float, default=0.1,
                        help='Portion of the data to use for test')
    parser.add_argument('--save_path', type=str, default='nyc/split')

    args = parser.parse_args()
    split(args.val_portion, args.test_portion, save_path=args.save_path)
