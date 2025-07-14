"""
Load models for prediction.
"""

import logging
import os
import pickle
from functools import lru_cache

import mlflow
import mlflow.artifacts
import mlflow.xgboost
from config import EXPERIMENT_NAME, RUN_ID, TRACKING_URI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
# Read from env variable
IS_MLFLOW = os.getenv("IS_MLFLOW", "False").lower() == "true"

if IS_MLFLOW:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


@lru_cache(maxsize=1)
def load_model():
    """
    Load model once and cache it
    """
    model_name = "xgboost_model"
    model_version = "latest"

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(f'{cur_dir}/model'):
        os.makedirs(f'{cur_dir}/model')

    model_path = f'{cur_dir}/model/model.b'

    if not os.path.exists(model_path):
        if IS_MLFLOW:
            model_uri = f"models:/{model_name}/{model_version}"
            model = mlflow.xgboost.load_model(model_uri)
            logger.info("Model loaded successfully from %s", model_uri)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            # raise exception
            raise Exception("Model not found")

    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    return model


@lru_cache(maxsize=1)
def load_preprocessor():
    """
    Load preprocessor once and cache it
    """

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(f'{cur_dir}/preprocessor'):
        os.makedirs(f'{cur_dir}/preprocessor')

    preprocessor_path = f'{cur_dir}/preprocessor/preprocessor.b'

    if not os.path.exists(preprocessor_path):

        # Download only if not exists
        if IS_MLFLOW:
            mlflow.artifacts.download_artifacts(
                artifact_uri=f'mlflow-artifacts:/1/{RUN_ID}/artifacts/preprocessor',
                dst_path=cur_dir,
            )
        else:
            raise Exception("Preprocessor not found")

    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)

    # Cache feature names

    logger.info("Preprocessor loaded successfully")

    return preprocessor
