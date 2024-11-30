# backend/Dockerfile.fastapi
FROM python:3.12-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8001

CMD ["python3", "app.py"]
