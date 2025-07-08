"""
Search for the best hyperparameters using Hyperopt.
"""
import os

import mlflow
import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi")


def objective(params):
    """
    Objective function for XGBoost model training.
    """
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(valid, 'validation')],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}


def search_best_params(search_space, max_evals: int =3):
    """
    Searches for the best hyperparameters using Bayesian Optimization
    """

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=Trials()
    )
    
    # convert to python dict, convert numpy arrays to lists, numpy floats to float
    best_result = {k: v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.float64) else v for k, v in best_result.items()}
    best_result['max_depth'] = int(best_result['max_depth'])
    return best_result


if __name__ == "__main__":

    # use argparse
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description='Train an XGBoost model.')
    
    parser.add_argument('--search_space_params_path', type=str,
                        default='search_space_params.yaml', help='Path to search space params')

    parser.add_argument('--xgb_matrices_path', type=str,
                        default='nyc/xgb_matrices', help='Path to the XGBoost matrices')
    parser.add_argument('--num_boost_round', type=int,
                        default=300, help='Number of boosting rounds')
    parser.add_argument('--early_stopping_rounds', type=int,
                        default=10, help='Early stopping rounds')

    parser.add_argument('--save_path', type=str,
                        default='nyc/best_train_params.yaml', help='Path to save the best parameters')

    args = parser.parse_args()

    # open the search space params file and load it into a dictionary
    with open(args.search_space_params_path, 'r', encoding='utf-8') as f:
        search_params = yaml.safe_load(f)

    distributions = {'quniform': hp.quniform, 'loguniform': hp.loguniform,
                     'lognormal': hp.lognormal, 'normal': hp.normal}

    max_depth_distr = distributions[search_params['max_depth']['distribution']]
    learning_rate_distr = distributions[search_params['learning_rate']
                                        ['distribution']]
    reg_alpha_distr = distributions[search_params['reg_alpha']['distribution']]
    reg_lambda_distr = distributions[search_params['reg_lambda']
                                     ['distribution']]
    min_child_weight_distr = distributions[search_params['min_child_weight']['distribution']]

    SEARCH_SPACE = {
        'max_depth': scope.int(max_depth_distr('max_depth', search_params['max_depth']['low'], search_params['max_depth']['high'], search_params['max_depth']['q'])),
        'learning_rate': learning_rate_distr('learning_rate', search_params['learning_rate']['low'], search_params['learning_rate']['high']),
        'reg_alpha': reg_alpha_distr('reg_alpha', search_params['reg_alpha']['low'], search_params['reg_alpha']['high']),
        'reg_lambda': reg_lambda_distr('reg_lambda', search_params['reg_lambda']['low'], search_params['reg_lambda']['high']),
        'min_child_weight': min_child_weight_distr('min_child_weight', search_params['min_child_weight']['low'], search_params['min_child_weight']['high']),
        'objective': search_params['objective'],
        'seed': search_params['seed']
    }

    xgb_matrices_path = args.xgb_matrices_path
    NUM_BOOST_ROUND = args.num_boost_round
    EARLY_STOPPING_ROUNDS = args.early_stopping_rounds

    # load X_train, X_val, X_test, y_train, y_val, y_test
    X_train = np.load(os.path.join(xgb_matrices_path,
                      'x_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(xgb_matrices_path,
                    'x_val.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(xgb_matrices_path, 'y_train.npy'))
    y_val = np.load(os.path.join(xgb_matrices_path, 'y_val.npy'))

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    best_params = search_best_params(search_space=SEARCH_SPACE, max_evals=search_params['max_evals'])

    # save best params
    with open(args.save_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(best_params, file)
