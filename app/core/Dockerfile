# backend/Dockerfile.django
FROM python:3.12-slim

WORKDIR /app

COPY core /app
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
