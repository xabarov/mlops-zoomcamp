import time

import requests

# Configuration
BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Test the root endpoint"""
    print("Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        print("✅ Root endpoint test passed\n")
    except Exception as e:
        print(f"❌ Root endpoint test failed: {e}\n")

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        print("✅ Health check test passed\n")
    except Exception as e:
        print(f"❌ Health check test failed: {e}\n")

def test_single_prediction():
    """Test single ride prediction"""
    print("Testing single prediction endpoint...")
    
    # Test data
    ride_data = {
        "day_of_week": "1",
        "hour_of_day": "12",
        "trip_distance": 3.5,
        "congestion_surcharge": 2.5,
        "passenger_count": 2
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=ride_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {result}")
        
        assert response.status_code == 200
        assert "predicted_duration" in result
        assert isinstance(result["predicted_duration"], (int, float))
        print("✅ Single prediction test passed\n")
        
        return result["predicted_duration"]
    except Exception as e:
        print(f"❌ Single prediction test failed: {e}\n")
        return None

def test_batch_prediction():
    """Test batch prediction"""
    print("Testing batch prediction endpoint...")
    
    # Test data with multiple rides
    batch_data = {
        "rides": [
            {
                "day_of_week": "1",
                "hour_of_day": "8",
                "trip_distance": 2.0,
                "congestion_surcharge": 2.5,
                "passenger_count": 1
            },
            {
                "day_of_week": "5",
                "hour_of_day": "18",
                "trip_distance": 5.5,
                "congestion_surcharge": 2.5,
                "passenger_count": 2
            },
            {
                "day_of_week": "6",
                "hour_of_day": "22",
                "trip_distance": 8.0,
                "congestion_surcharge": 0.0,
                "passenger_count": 3
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {result}")
        
        assert response.status_code == 200
        assert "predictions" in result
        assert len(result["predictions"]) == len(batch_data["rides"])
        assert all(isinstance(pred, (int, float)) for pred in result["predictions"])
        print("✅ Batch prediction test passed\n")
        
        return result["predictions"]
    except Exception as e:
        print(f"❌ Batch prediction test failed: {e}\n")
        return None

def test_invalid_data():
    """Test with invalid data to check error handling"""
    print("Testing error handling with invalid data...")
    
    # Missing required field
    invalid_data = {
        "day_of_week": "1",
        "hour_of_day": "12",
        "trip_distance": 3.5
        # Missing congestion_surcharge and passenger_count
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        assert response.status_code == 422  # Validation error
        print("✅ Invalid data test passed (correctly rejected)\n")
    except Exception as e:
        print(f"❌ Invalid data test failed: {e}\n")

def test_performance():
    """Test response time performance"""
    print("Testing performance...")
    
    ride_data = {
        "day_of_week": "1",
        "hour_of_day": "12",
        "trip_distance": 3.5,
        "congestion_surcharge": 2.5,
        "passenger_count": 2
    }
    
    times = []
    num_requests = 10
    
    for i in range(num_requests):
        start_time = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=ride_data,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"Performance results ({len(times)} successful requests):")
        print(f"  Average response time: {avg_time:.3f}s")
        print(f"  Min response time: {min_time:.3f}s")
        print(f"  Max response time: {max_time:.3f}s")
        print("✅ Performance test completed\n")
    else:
        print("❌ No successful requests for performance test\n")

def test_different_scenarios():
    """Test various realistic scenarios"""
    print("Testing different ride scenarios...")
    
    scenarios = [
        {
            "name": "Morning rush hour",
            "data": {
                "day_of_week": "1",  # Monday
                "hour_of_day": "8",
                "trip_distance": 6.0,
                "congestion_surcharge": 2.5,
                "passenger_count": 1
            }
        },
        {
            "name": "Evening rush hour",
            "data": {
                "day_of_week": "5",  # Friday
                "hour_of_day": "18",
                "trip_distance": 6.0,
                "congestion_surcharge": 2.5,
                "passenger_count": 2
            }
        },
        {
            "name": "Weekend night",
            "data": {
                "day_of_week": "6",  # Saturday
                "hour_of_day": "23",
                "trip_distance": 6.0,
                "congestion_surcharge": 2.5,
                "passenger_count": 4
            }
        },
        {
            "name": "Long distance trip",
            "data": {
                "day_of_week": "3",  # Wednesday
                "hour_of_day": "14",
                "trip_distance": 60.0,
                "congestion_surcharge": 2.5,
                "passenger_count": 1
            }
        }
    ]
    
    for scenario in scenarios:
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=scenario["data"],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                duration = response.json()["predicted_duration"]
                print(f"  {scenario['name']}: {duration:.2f} minutes")
            else:
                print(f"  {scenario['name']}: Failed with status {response.status_code}")
                
        except Exception as e:
            print(f"  {scenario['name']}: Error - {e}")
    
    print("✅ Scenario testing completed\n")

def main():
    """Run all tests"""
    print("Starting API tests...\n" + "="*50)
    
    # Check if server is running
    try:
        response = requests.get(BASE_URL, timeout=5)
        print("✅ Server is running\n")
    except requests.exceptions.RequestException:
        print("❌ Server is not running. Please start the FastAPI server first.")
        print("Run: python predict.py")
        return
    
    # Run all tests
    test_root_endpoint()
    test_health_check()
    test_single_prediction()
    test_batch_prediction()
    test_invalid_data()
    test_performance()
    test_different_scenarios()
    
    print("="*50)
    print("All tests completed!")

if __name__ == "__main__":
    main()