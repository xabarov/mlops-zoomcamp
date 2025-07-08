"""
Train an XGBoost model on the NYC taxi
"""
import json
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import seaborn as sns
import xgboost as xgb
import yaml
from mlflow.models.signature import infer_signature
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi")


def comprehensive_feature_importance_analysis(model):
    """Analyze and log comprehensive feature importance."""

    # Get importance scores
    importance = model.get_score(importance_type='weight')

    # Sort features by importance
    sorted_features = sorted(
        importance.items(), key=lambda x: x[1], reverse=True
    )

    # Create visualization
    features, scores = zip(*sorted_features[:10])

    plt.figure(figsize=(10, 8))
    sns.barplot(x=list(scores), y=list(features))
    plt.title("Top 10 Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()

    # Save and log plot
    plot_filename = "nyc/feature_importance.png"
    plt.savefig(plot_filename, bbox_inches="tight")
    mlflow.log_artifact(plot_filename)
    plt.close()

    # Log importance as JSON artifact
    json_filename = "nyc/feature_importance.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(importance, f, indent=2)
    mlflow.log_artifact(json_filename)


def train_best_model(best_params: dict, xgb_matrices_path: str,
                     num_boost_round: int = 300,
                     early_stopping_rounds: int = 30,
                     features_path: str = 'features.yaml', 
                     save_model_path: str = 'nyc/models/booster.json'):
    """
    Train the best model using the provided parameters and log metrics to MLflow.
    """

    # load X_train, X_val, X_test, y_train, y_val, y_test
    X_train = np.load(os.path.join(xgb_matrices_path,
                      'x_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(xgb_matrices_path,
                    'x_val.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(xgb_matrices_path, 'y_train.npy'))
    y_val = np.load(os.path.join(xgb_matrices_path, 'y_val.npy'))

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    # read categorical and numerical features from features.yaml
    with open(features_path, 'r', encoding='utf-8') as f:
        features = yaml.safe_load(f)['features']
        categorical = features['categorical']
        numerical = features['numerical']

    with open('params.yaml', 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)['dataset']
        year = params['year']
        month = params['month']

    # if models folder does not exist, create it
    if not os.path.exists("nyc/models"):
        os.makedirs("nyc/models")

    mlflow.end_run()

    with mlflow.start_run():

        mlflow.log_params(best_params)

        booster = xgb.train(best_params, dtrain=train,
                            num_boost_round=num_boost_round,
                            evals=[(valid, 'validation')],
                            early_stopping_rounds=early_stopping_rounds
                            )

        y_pred = booster.predict(valid)

        # get y_val from valid dmatrix
        y_val = valid.get_label()

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        mlflow.log_param('categorical_features', categorical)
        mlflow.log_param('numerical_features', numerical)
        mlflow.log_param('yellow_taxi_dataset_year', year)
        mlflow.log_param('yellow_taxi_dataset_month', month)
        mlflow.log_param('num_boost_round', num_boost_round)
        mlflow.log_param('early_stopping_rounds', early_stopping_rounds)

        comprehensive_feature_importance_analysis(booster)

        mlflow.log_param('model', 'XGBoost')

        mlflow.log_artifact('nyc/models/preprocessor.b',
                            artifact_path="preprocessor")

        signature = infer_signature(train.get_data(), y_pred)

        mlflow.xgboost.log_model(booster, artifact_path="model",     signature=signature,
                                 input_example=X_train[:5],
                                 registered_model_name="xgboost_model")

        # save model to nyc/models path
        booster.save_model(save_model_path)


if __name__ == "__main__":

    # use argparse
    import argparse

    parser = argparse.ArgumentParser(description='Train an XGBoost model.')

    parser.add_argument('--best_params_path', type=str,
                        required=True, help='Path to the best parameters file')
    parser.add_argument('--xgb_matrices_path', type=str,
                        default='nyc/xgb_matrices', help='Path to the XGBoost matrices')
    parser.add_argument('--num_boost_round', type=int,
                        default=300, help='Number of boosting rounds')
    parser.add_argument('--early_stopping_rounds', type=int,
                        default=10, help='Early stopping rounds')

    parser.add_argument('--features_path', type=str,
                        default='features.yaml', help='Path to the features file')

    parser.add_argument('--save_model_path', type=str,
                        default='nyc/models/model.json', help='Path to save the model')

    args = parser.parse_args()

    # load best params
    with open(args.best_params_path, 'r', encoding='utf-8') as file:
        best_params = yaml.safe_load(file)

    train_best_model(best_params, args.xgb_matrices_path,
                     int(args.num_boost_round), int(args.early_stopping_rounds),
                     args.features_path, save_model_path=args.save_model_path)
