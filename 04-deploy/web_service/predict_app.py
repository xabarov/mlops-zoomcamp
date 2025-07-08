"""
Predict taxi fares using a trained XGBoost model
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from loaders import load_model, load_preprocessor
from models import (BatchPredictionResponse, BatchRideData, HealthResponse,
                    PredictionResponse, RideData)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the ML model and preprocessor once and cache
    """
    # Load the ML model
    ml_models["xgboost_model"] = load_model()
    ml_models["preprocessor"] = load_preprocessor()

    logger.info("All models initialized")
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


# FastAPI app
app = FastAPI(
    title="NYC Taxi Duration Prediction API",
    description="API for predicting NYC taxi ride durations",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", summary="Root endpoint")
async def root():
    """
    Root endpoint for the API. Returns a welcome message
    """
    return {"message": "NYC Taxi Duration Prediction API", "version": "1.0.0"}


@app.post("/predict", summary="Predict single ride duration")
async def predict_duration(ride_data: RideData) -> PredictionResponse:
    """
    Predict ride duration for a single ride
    """
    try:
        # Convert Pydantic model to dict
        ride_dict = ride_data.model_dump()

        x_transformed = ml_models["preprocessor"].transform(ride_dict)
        x_dmatrix = xgb.DMatrix(x_transformed)
        prediction = ml_models["xgboost_model"].predict(x_dmatrix)

        return PredictionResponse(predicted_duration=float(prediction[0]))

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", summary="Predict multiple ride durations")
async def predict_batch(batch_data: BatchRideData) -> BatchPredictionResponse:
    """
    Predict ride duration for multiple rides (batch processing)
    """
    try:

        # Convert Pydantic models to dict list
        rides_dict = [ride.model_dump() for ride in batch_data.rides]

        # Transform batch data
        x_transformed = ml_models['preprocessor'].transform(rides_dict)

        # Create DMatrix
        x_dmatrix = xgb.DMatrix(x_transformed)

        # Predict
        predictions = ml_models["xgboost_model"].predict(x_dmatrix)

        return BatchPredictionResponse(predictions=[float(pred) for pred in predictions])

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/health", summary="Health check endpoint")
async def health_check() -> HealthResponse:
    """Check if models are loaded and working"""
    try:
        test_ride = RideData(
            day_of_week="1",
            hour_of_day="12",
            trip_distance=3.0,
            congestion_surcharge=1.0,
            passenger_count=1
        )
        await predict_duration(test_ride)
        return HealthResponse(status="healthy", models_loaded=True)
    except Exception as e:
        return HealthResponse(status="unhealthy", models_loaded=False, error=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "predict_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
