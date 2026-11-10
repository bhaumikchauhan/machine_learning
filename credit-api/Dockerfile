FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app
COPY predictor.py /app
COPY schemas.py /app
COPY model.joblib /app

EXPOSE 8000

CMD ["fastapi", "run", "app.py"]