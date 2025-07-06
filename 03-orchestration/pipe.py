import json
import os
import pickle

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi")

# if models folder does not exist, create it
if not os.path.exists("models"):
    os.makedirs("models")


def read_dataframe(year, month):
    """
    Reads the NYC taxi dataset from a parquet
    """

    if not os.path.exists(f"yellow_tripdata_{year}-{month}.parquet"):
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet'
    else:
        url = f'yellow_tripdata_{year}-{month}.parquet'

    df = pd.read_parquet(url)

    return df


def feature_engineering(df):
    """
    Feature engineering for the NYC taxi dataset.
    """

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # extract day of week and hour of day and put into new columns
    df['day_of_week'] = df.tpep_pickup_datetime.dt.dayofweek
    df['hour_of_day'] = df.tpep_pickup_datetime.dt.hour
    df['day_of_week'] = df['day_of_week'].astype(str)
    df['hour_of_day'] = df['hour_of_day'].astype(str)

    # get 'congestion_surcharge', 'fare_amount', 'tip_amount', 'total_amount'
    # and convert to float, delete rows with null values
    for field in ['congestion_surcharge', 'fare_amount', 'tip_amount', 'total_amount']:
        # df[field] = pd.to_numeric(df[field], errors='coerce')
        df = df[df[field].notna()]
        df[field] = df[field].astype(float)

    return df


def split_data(df):

    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [
                                         int(.8*len(df)), int(.9*len(df))])

    return df_train, df_val, df_test


def create_pipe(categorical, numerical):

    ohe = OneHotEncoder(handle_unknown='ignore')

    full_pipeline = ColumnTransformer(
        transformers=[
            ('ohe', ohe, categorical),
            ('scaler', StandardScaler(), numerical),
        ],
        verbose_feature_names_out=False,  # Ensure short feature names
        n_jobs=-1
    )

    return full_pipeline


def transform_data(df_train, df_val, df_test, full_pipeline, categorical, numerical):

    # transform the training and validation data using the full pipeline
    X_train = full_pipeline.fit_transform(df_train[categorical + numerical])
    X_val = full_pipeline.transform(df_val[categorical + numerical])
    X_test = full_pipeline.transform(df_test[categorical + numerical])

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_dmatrix(X_train, X_val, X_test, y_train, y_val, y_test):

    train = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    valid = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    test = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    return train, valid, test


def objective(params):
    """
    Objective function for XGBoost model training.
    """
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)

        mlflow.log_param('categorical_features', CATEGORICAL)
        mlflow.log_param('numerical_features', NUMERICAL)

        mlflow.log_param('dataset_year', YEAR)
        mlflow.log_param('dataset_month', MONTH)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}


def search_best_params(search_space):
    """
    Searches for the best hyperparameters using Bayesian Optimization
    """

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=3,
        trials=Trials()
    )

    return best_result


def comprehensive_feature_importance_analysis(model):
    """Analyze and log comprehensive feature importance."""

    importance_types = ["weight", "gain", "cover", "total_gain"]

    for imp_type in importance_types:
        # Get importance scores
        importance = model.get_score(importance_type=imp_type)

        if not importance:
            continue

        # Sort features by importance
        sorted_features = sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )

        # Create visualization
        features, scores = zip(*sorted_features[:10])

        plt.figure(figsize=(10, 8))
        sns.barplot(x=list(scores), y=list(features))
        plt.title(f"Top 10 Feature Importance ({imp_type.title()})")
        plt.xlabel("Importance Score")
        plt.tight_layout()

        # Save and log plot
        plot_filename = f"feature_importance_{imp_type}.png"
        plt.savefig(plot_filename, bbox_inches="tight")
        mlflow.log_artifact(plot_filename)
        plt.close()

        # Log importance as JSON artifact
        json_filename = f"feature_importance_{imp_type}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(importance, f, indent=2)
        mlflow.log_artifact(json_filename)


def train_best_model(best_params):

    mlflow.end_run()

    with mlflow.start_run():

        mlflow.log_params(best_params)

        booster = xgb.train(best_params, dtrain=train,
                            num_boost_round=1000,
                            evals=[(valid, 'validation')],
                            early_stopping_rounds=50
                            )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        mlflow.log_param('categorical_features', CATEGORICAL)
        mlflow.log_param('numerical_features', NUMERICAL)

        comprehensive_feature_importance_analysis(booster)

        mlflow.log_param('model', 'XGBoost')

        with open('models/preprocessor.b', 'wb') as f_out:
            pickle.dump(pipe, f_out)

        mlflow.log_artifact('models/preprocessor.b',
                            artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="model")


if __name__ == "__main__":

    YEAR = 2025
    MONTH = '01'

    CATEGORICAL = ['day_of_week', 'hour_of_day']
    NUMERICAL = ['trip_distance', 'congestion_surcharge']

    SEARCH_SPACE = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    BEST_PARAMS = {
        'learning_rate': 0.06836426267409443,
        'max_depth': 10,
        'min_child_weight': 14.354365207007865,
        'objective': 'reg:linear',
        'reg_alpha': 0.2042301820266,
        'reg_lambda': 0.11861308163,
        'seed': 42,
    }

    df = read_dataframe(year=YEAR, month=MONTH)
    df = feature_engineering(df)

    df_train, df_val, df_test = split_data(df)

    pipe = create_pipe(CATEGORICAL, NUMERICAL)

    X_train, X_val, X_test, y_train, y_val, y_test = transform_data(
        df_train, df_val, df_test, pipe, CATEGORICAL, NUMERICAL)

    feature_names = list(pipe.get_feature_names_out())

    train, valid, test = create_dmatrix(
        X_train, X_val, X_test, y_train, y_val, y_test)

    # best_params = search_best_params(SEARCH_SPACE)

    train_best_model(BEST_PARAMS)
