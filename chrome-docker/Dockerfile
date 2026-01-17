FROM selenium/standalone-chrome:latest

# Switch to root for setup
USER root

# Install Python and Selenium
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install selenium webdriver-manager

# Create extension directory
RUN mkdir -p /opt/extension

# Copy extension files
COPY ./browser-extension /opt/extension
COPY ./chrome-docker/automator.py /app/automator.py

# Set permissions
RUN chmod -R 777 /opt/extension /app

# Chrome options to load extension
ENV CHROME_OPTS="--load-extension=/opt/extension --no-sandbox --disable-dev-shm-usage"

# Set entrypoint to run automation script (or keep standard entrypoint and run script in background)
# For now, let's keep the standard entrypoint so the hub starts, and we'll run automator separately.
# Or we can customize the start script.

WORKDIR /app

# Switch back to seluser
USER 1200

# Startup script
RUN echo '#!/bin/bash\n/opt/bin/entry_point.sh &\nsleep 10\npython3 /app/automator.py\nwait' > /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
