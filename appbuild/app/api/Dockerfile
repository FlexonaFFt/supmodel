# backend/Dockerfile.fastapi
FROM python:3.12-slim

WORKDIR /app

COPY backend/api /app
RUN pip install -r requirements.txt

EXPOSE 8001

CMD ["python3", "app.py"]
