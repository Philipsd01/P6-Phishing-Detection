FROM python:3.12.9

WORKDIR /app

# Install system dependencies if needed:

# RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files into the container
COPY . /app

# Expose the port your Flask app uses
EXPOSE 5000

# Run the API server
CMD ["python", "server/api_server.py"]