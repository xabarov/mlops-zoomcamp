
"""
Transform the yellow taxi dataset for use in machine learning
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def run(categorical: list[str], numerical: list[str],
        save_path: str = 'nyc/xgb_matrices'):
    """
    Run the data transformation pipeline
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # read from
    df_train = pd.read_parquet('nyc/split/train.parquet')
    df_val = pd.read_parquet('nyc/split/val.parquet')
    df_test = pd.read_parquet('nyc/split/test.parquet')
    
    pipe = DictVectorizer(sparse=False)

    X_train = pipe.fit_transform(df_train[categorical + numerical].to_dict(orient='records'))
    X_val = pipe.transform(df_val[categorical + numerical].to_dict(orient='records'))
    X_test = pipe.transform(df_test[categorical + numerical].to_dict(orient='records'))
    
    # Convert sparse matrices to dense arrays
    X_train = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    X_val = X_val.toarray() if hasattr(X_val, 'toarray') else X_val
    X_test = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values


    # save train, valid and test data to pickle files
    np.save(os.path.join(save_path, 'x_train.npy'), X_train, allow_pickle=True)
    np.save(os.path.join(save_path, 'x_val.npy'), X_val, allow_pickle=True)
    np.save(os.path.join(save_path, 'x_test.npy'), X_test, allow_pickle=True)
    
    # save y_train, y_val and y_test to numpy files
    np.save(os.path.join(save_path, 'y_train.npy'), y_train)
    np.save(os.path.join(save_path, 'y_val.npy'), y_val)
    np.save(os.path.join(save_path, 'y_test.npy'), y_test)

    if not os.path.exists('nyc/models'):
        os.makedirs('nyc/models')

    with open('nyc/models/preprocessor.b', 'wb') as f_out:
        pickle.dump(pipe, f_out)


if __name__ == "__main__":

    import argparse

    import yaml

    parser = argparse.ArgumentParser(description='Transform data for XGBoost')
    parser.add_argument('--features_path', type=str,
                        default='features.yaml', help='Path to the features.yaml file')
    parser.add_argument('--save_path', type=str, default='nyc/xgb_matrices',
                        help='Path to save the transformed data')

    args = parser.parse_args()

    # read features.yaml
    with open(args.features_path, 'r', encoding='utf-8') as f:
        features = yaml.safe_load(f)['features']

    run(categorical=features['categorical'],
        numerical=features['numerical'], save_path=args.save_path)
