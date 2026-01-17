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
ARG CACHEBUST=WHALE_FIX_1768677000
RUN echo "===================" && echo "BUILD: $CACHEBUST" && date && echo "==================="
COPY . .

# Expose port
EXPOSE 8050

# Environment variables
ENV PYTHONUNBUFFERED=1

# Run the application with Gunicorn - optimized for stability
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--threads", "4", "--timeout", "300", "--keep-alive", "5", "--graceful-timeout", "120", "realtime_dashboard:server"]
