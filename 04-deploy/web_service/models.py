from pydantic import BaseModel

# Pydantic models for request/response


class RideData(BaseModel):
    """
    Ride data for prediction.
    """
    day_of_week: str
    hour_of_day: str
    trip_distance: float
    congestion_surcharge: float
    passenger_count: int


class BatchRideData(BaseModel):
    """
    Batch ride data for prediction.
    """
    rides: list[RideData]


class PredictionResponse(BaseModel):
    """
    Prediction response for a single ride.
    """
    predicted_duration: float


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response for multiple rides.
    """
    predictions: list[float]


class HealthResponse(BaseModel):
    """
    Health response for the service.
    """
    status: str
    models_loaded: bool
    error: str = None