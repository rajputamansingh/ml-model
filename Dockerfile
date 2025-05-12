FROM python:3.10-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run setup script to create sample images
RUN python setup.py

# Create directory for Streamlit config
RUN mkdir -p .streamlit

# Copy Streamlit config if it exists
RUN if [ -f "config.toml" ]; then cp config.toml .streamlit/config.toml; fi

# Expose port
EXPOSE 8501

# Command to run on container start
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
