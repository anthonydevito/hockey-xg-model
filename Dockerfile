# Using a slim version of Python for faster builds
FROM python:3.11-slim

# Directory setup
WORKDIR /app

# Copy & install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy src code
COPY src/ ./src/

# Actual command to run your training script
CMD ["python", "src/train.py"]