quality-check:
	isort .
	black .
	pylint 04-deploy/web_service

unit-test: quality-check
	pytest tests/units

build:
	cd 04-deploy/web_service && docker build -t nyc-predictor .


integration-test: build
	docker run -d -p 8000:8000 -e IS_MLFLOW=False nyc-predictor:latest ; \
	CONTAINER_ID=$$(docker ps -q) ; \
	echo $${CONTAINER_ID} ; \
	sleep 2 ; \
	pytest tests/test_web_service.py ; \
	docker stop $${CONTAINER_ID}

setup:
	pip install -r requirements.txt
	pre-commit install
