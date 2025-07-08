"""
Download the NYC taxi dataset for a given year and month.
"""
import os
import pandas as pd


def download(year: str, month: str, save_path: str = 'nyc/yellow_tripdata.parquet'):
    """
    Download the NYC taxi dataset for a given year and month.
    """

    if not os.path.exists("nyc"):
        os.makedirs("nyc")

    if not os.path.exists(save_path):
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet'
        df = pd.read_parquet(url)
        # save locally
        df.to_parquet(save_path)


if __name__ == "__main__":
    # argparse to get year and month from command line
    import argparse

    parser = argparse.ArgumentParser(description='Download NYC taxi dataset')
    parser.add_argument('--year', type=str, required=True)
    parser.add_argument('--month', type=str, required=True)
    parser.add_argument('--save_path', type=str,
                        default='nyc/yellow_tripdata.parquet')

    args = parser.parse_args()

    download(args.year, args.month, save_path=args.save_path)
