# Start from a small Linux image that already has Python 3.13
FROM python:3.13-slim

# All following commands run inside /app in the container
WORKDIR /app

# Install the libraries first (Docker caches this step, so rebuilds are fast)
COPY A1/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the app needs: the code and the saved model
COPY A1/app_a1.py A1/app_a1.py
COPY A1/models_a1 A1/models_a1

# The app listens on port 8050
EXPOSE 8050

# Start the app with gunicorn (a production web server; "server" is defined in app_a1.py)
CMD ["gunicorn", "--chdir", "A1", "--bind", "0.0.0.0:8050", "--workers", "2", "app_a1:server"]