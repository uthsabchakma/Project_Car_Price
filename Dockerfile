FROM python:3.13-slim
WORKDIR /app

COPY A1/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the app needs: A1's model (for the old page) and A2's code + model
COPY A1/models_a1 A1/models_a1
COPY A2/app_a2.py A2/linear_regression.py A2/
COPY A2/models_a2 A2/models_a2

EXPOSE 8050

# Start the two-page app (A2) with gunicorn
CMD ["gunicorn", "--chdir", "A2", "--bind", "0.0.0.0:8050", "--workers", "2", "app_a2:server"]