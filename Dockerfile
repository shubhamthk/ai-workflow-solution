FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src ./src
COPY eda ./eda
COPY run_tests.py .
ENV PYTHONPATH=/app

# Build artifacts (train best model) at image build-time
RUN python -c "from src.train import train_and_compare; train_and_compare()"

# Expose API
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
