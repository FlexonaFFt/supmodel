# backend/core/Dockerfile

# Используем Python в качестве базового образа
FROM python:3.10

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код Django
COPY . .

# Экспортируем порт
EXPOSE 8000

# Запускаем Django сервер
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
