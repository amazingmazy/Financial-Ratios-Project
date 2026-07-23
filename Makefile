.PHONY: help install install-dev lint test test-coverage format clean docker-build docker-run docker-stop docker-clean all

# Default target
help:
	@echo "Available targets:"
	@echo "  install          - Install production dependencies"
	@echo "  install-dev      - Install development dependencies"
	@echo "  lint             - Run linting (pylint + flake8)"
	@echo "  format           - Format code with black"
	@echo "  test             - Run tests"
	@echo "  test-coverage    - Run tests with coverage"
	@echo "  docker-build     - Build Docker image"
	@echo "  docker-run       - Run Docker container"
	@echo "  docker-stop      - Stop Docker container"
	@echo "  docker-clean     - Remove Docker image and container"
	@echo "  clean            - Clean up cache and temp files"
	@echo "  all              - Run lint, test, and docker-build"

# Install dependencies
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# Linting
lint:
	@echo "Running pylint..."
	pylint app/ tests/ --disable=C0111,R0903,W0107
	@echo "Running flake8..."
	flake8 app/ tests/ --max-line-length=100 --ignore=E203,W503

# Formatting
format:
	@echo "Formatting code with black..."
	black app/ tests/ --line-length=100

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t stock-prediction-api:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -d --name stock-prediction-api -p 8000:8000 stock-prediction-api:latest
	@echo "API running at http://localhost:8000"
	@echo "Docs at http://localhost:8000/docs"

docker-stop:
	@echo "Stopping Docker container..."
	docker stop stock-prediction-api
	docker rm stock-prediction-api

docker-clean:
	@echo "Cleaning up Docker resources..."
	docker stop stock-prediction-api || true
	docker rm stock-prediction-api || true
	docker rmi stock-prediction-api:latest || true

# Docker Compose commands
compose-up:
	@echo "Starting services with Docker Compose..."
	docker-compose up -d
	@echo "API running at http://localhost:8000"

compose-down:
	@echo "Stopping services..."
	docker-compose down

compose-logs:
	docker-compose logs -f

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true

# Run all checks
all: lint test docker-build
	@echo "All checks passed!"