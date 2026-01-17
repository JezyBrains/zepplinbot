# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed for numpy/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
# AGGRESSIVE cache buster - completely different value each time
ARG CACHEBUST=NOCACHE_1768672901
RUN echo "===================" && echo "BUILD: $CACHEBUST" && date && echo "==================="
COPY . .

# Expose port
EXPOSE 8050

# Environment variables
ENV PYTHONUNBUFFERED=1

# Run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "1", "--threads", "4", "--timeout", "120", "realtime_dashboard:server"]
