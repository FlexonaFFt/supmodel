# frontend/Dockerfile

# Используем Node.js для установки и запуска Parcel
FROM node:18

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем package.json и устанавливаем зависимости
COPY package.json .
RUN npm install

# Копируем исходный код
COPY . .

# Экспортируем порт для Parcel
EXPOSE 1234

# Команда для запуска приложения с Parcel
CMD ["npx", "parcel", "pages/checkout/checkout.html", "--host", "0.0.0.0", "--port", "1234"]
