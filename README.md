# MLOps Zoomcamp

This repository contains materials and code for the MLOps Zoomcamp course, focusing on machine learning operations, experiment tracking, model registry, and deployment practices.

## Project Structure

### 01-intro/
Contains Jupyter notebook for train model for NYC taxi dataset, that will be used in this course.

### 02-experiment-tracking/
Contains Jupyter notebooks for experiment tracking and model registry examples:

### 03-orchestration/
Contains python pipe.py for orchestrating machine learning pipeline, later will use to convert to DVC DAG

### 04-deploy/web_service/
Production deployment code for serving ML models:
- web_service - FastAPI web service for serving ML models
- batch_service - Batch prediction service using MLflow models
- streaming - Streaming type of service using MLflow models

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start MLflow tracking server:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

3. Download data:

```bash
python download_data.py --year 2025 --month 01
```

It will download the NYC taxi dataset for the year 2025 and month 01 and save it in `data/yellow_tripdata.parquet`

4. Check DVC DAG:

Run this to show the DVC DAG:
```bash
dvc dag
```
Check this yaml configs:
- **params.yaml** - contains parameters for data download, feature engineering, split, transformation and model training.
- **features.yaml** - contains information about categorical and numerical features and target variable.
- **search_space_params.yaml** - contains parameters for hyperparameter tuning.
- **dvc.yaml** - defines the data versioning and pipeline steps.

4. Run DVC pipeline:

Run following command to reproduce the pipeline:

```bash
dvc repro
```