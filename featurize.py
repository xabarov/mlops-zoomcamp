"""
Feature engineering for the NYC taxi dataset.
"""
import os

import pandas as pd


def feature_engineering(features: dict,
                        dataset_path: str = 'data/raw/yellow_tripdata.parquet',
                        duration_minutes_min: float = 1,
                        duration_minutes_max: float = 60,
                        save_path: str = 'nyc/processed/yellow_tripdata_processed.parquet'):
    """
    Feature engineering for the NYC taxi dataset.
    """

    if not os.path.exists("nyc/processed"):
        os.makedirs("nyc/processed")

    df = pd.read_parquet(dataset_path)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= duration_minutes_min) &
            (df.duration <= duration_minutes_max)]

    # extract day of week and hour of day and put into new columns
    df['day_of_week'] = df.tpep_pickup_datetime.dt.dayofweek
    df['hour_of_day'] = df.tpep_pickup_datetime.dt.hour
    df['day_of_week'] = df['day_of_week'].astype(str)
    df['hour_of_day'] = df['hour_of_day'].astype(str)

    # get 'congestion_surcharge', 'fare_amount', 'tip_amount', 'total_amount'
    # and convert to float, delete rows with null values
    for field in features['numerical']:
        df = df[df[field].notna()]
        df[field] = df[field].astype(float)

    # drop columns that are not in features['categorical'] or ['duration'] or ['numerical']
    drop_columns = [feature for feature in df.columns if feature not in features['categorical'] + ['duration'] + features['numerical']]
    df.drop(columns=drop_columns, inplace=True)

    # save to parquet
    df.to_parquet(save_path)


if __name__ == "__main__":
    # argparse to params for feature engineering
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description='Feature engineering for NYC taxi dataset')
    
    parser.add_argument('--dataset_path', type=str, default='data/raw/yellow_tripdata.parquet')
    parser.add_argument('--duration_minutes_min', type=float, default=1.0)
    parser.add_argument('--duration_minutes_max', type=float, default=60.0)
    parser.add_argument('--features_path', type=str,
                        default='features.yaml', help='Path to the features.yaml file')

    parser.add_argument('--save_path', type=str,
                        default='nyc/processed/yellow_tripdata_processed.parquet')

    args = parser.parse_args()

    # read features.yaml
    with open(args.features_path, 'r', encoding='utf-8') as f:
        features = yaml.safe_load(f)['features']

    feature_engineering(features,
                        dataset_path=args.dataset_path,
                        duration_minutes_min=args.duration_minutes_min,
                        duration_minutes_max=args.duration_minutes_max,
                        save_path=args.save_path)
