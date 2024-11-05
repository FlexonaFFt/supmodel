# backend/api/Dockerfile

# Используем Python в качестве базового образа
FROM python:3.10

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код FastAPI
COPY . .

# Экспортируем порт
EXPOSE 8001

# Запускаем FastAPI с Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
