FROM python:3.13-slim
WORKDIR /app
COPY A1/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY A1/models_a1 A1/models_a1
COPY A2/linear_regression.py A2/
COPY A2/models_a2 A2/models_a2
COPY A3/app_a3.py A3/logistic_regression.py A3/
COPY A3/models_a3 A3/models_a3
EXPOSE 8050
CMD ["gunicorn", "--chdir", "A3", "--bind", "0.0.0.0:8050", "--workers", "2", "app_a3:server"]
