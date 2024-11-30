# backend/Dockerfile.fastapi
FROM python:3.12-slim

WORKDIR /app

COPY api /app
RUN pip install -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "api.model:app", "--host", "0.0.0.0", "--port", "8001"]
