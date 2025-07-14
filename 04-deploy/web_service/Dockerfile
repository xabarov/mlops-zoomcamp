# Use a Python base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

# Copy the web service directory into the container
COPY . /app

# Expose the port your application runs on
EXPOSE 8000

# Command to run your application (e.g., using Uvicorn for a FastAPI app)
CMD ["uvicorn", "predict_app:app", "--host", "0.0.0.0", "--port", "8000"]
