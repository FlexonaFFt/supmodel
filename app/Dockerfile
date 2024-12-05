FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements.txt
COPY db.sqlite3 db.sqlite3
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# docker build -t startup-ml-model .
# docker run -p 8000:8000 startup-ml-model
