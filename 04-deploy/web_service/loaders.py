import logging
import os
import pickle
from functools import lru_cache

import mlflow
import mlflow.artifacts
import mlflow.xgboost
from config import RUN_ID, TRACKING_URI, EXPERIMENT_NAME
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


@lru_cache(maxsize=1)
def load_model():
    """
    Load model once and cache it
    """
    model_name = "xgboost_model"
    model_version = "latest"

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.xgboost.load_model(model_uri)
    logger.info(f"Model loaded successfully from {model_uri}")

    return model


@lru_cache(maxsize=1)
def load_preprocessor():
    """
    Load preprocessor once and cache it
    """

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessor_path = f'{cur_dir}/preprocessor/preprocessor.b'

    # Download only if not exists
    mlflow.artifacts.download_artifacts(
        artifact_uri=f'mlflow-artifacts:/1/{RUN_ID}/artifacts/preprocessor',
        dst_path=cur_dir
    )

    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)

    # Cache feature names

    logger.info("Preprocessor loaded successfully")

    return preprocessor
