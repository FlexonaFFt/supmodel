# notebook


## Шаги для создания предсказательной модели:

1. Чтение и обработка данных: Мы будем читать данные из файла и обрабатывать их с помощью библиотеки pandas, а затем нормализовать или закодировать числовые и категориальные данные.
2. Определение целевой переменной: Нам нужно определить, какую переменную будем предсказывать. Например, это может быть показатель прибыли (Projected_Profit_1Y) или успех на основе совокупных показателей.
3. Создание и обучение модели: Мы можем попробовать регрессионные модели или классификацию, если нужно предсказывать категорию успеха.

Пример кода
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Загрузка данных
data = pd.DataFrame({
    'ID': [1, 2, 3],  # Пример данных
    'Category_Code': [2, 3, 1],
    'Market_Demand_Index': [75, 60, 80],
    'Initial_Budget': [500000, 300000, 800000],
    'Projected_Profit_1Y': [200000, 150000, 250000],
    'Number_of_Competitors': [3, 5, 2],
    'Competitive_Index': [85, 70, 90]
})

# Целевая переменная (успешность стартапа)
target = 'Projected_Profit_1Y'

# Разделяем на признаки и целевую переменную
X = data.drop(columns=[target])
y = data[target]

# Разделение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Модель случайного леса
model = RandomForestRegressor()

# Обучение модели
model.fit(X_train_scaled, y_train)

# Предсказания
y_pred = model.predict(X_test_scaled)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

## Напишем базовую архитектуру модели

Мы будем использовать нейронную сеть с несколькими слоями, обучающуюся на данных стартапов для предсказания, например, прибыли через год. Для этого можно использовать PyTorch.

### Шаги написания модели на PyTorch:

1. Определение нейронной сети: Мы создаём архитектуру модели — это набор слоёв, которые будут обучаться на наших данных.
2. Определение функции потерь: Для регрессионных задач мы обычно используем среднеквадратичную ошибку.
3. Определение оптимизатора: Оптимизатор (например, Adam) обновляет веса модели, минимизируя ошибку.

```
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Пример данных, которые нужно будет заменить на реальные
data = pd.DataFrame({
    'ID': [1, 2, 3],
    'Category_Code': [2, 3, 1],
    'Market_Demand_Index': [75, 60, 80],
    'Initial_Budget': [500000, 300000, 800000],
    'Projected_Profit_1Y': [200000, 150000, 250000],
    'Number_of_Competitors': [3, 5, 2],
    'Competitive_Index': [85, 70, 90]
})

# Предположим, что 'Projected_Profit_1Y' - это целевая переменная
X = data.drop(columns=['Projected_Profit_1Y'])
y = data['Projected_Profit_1Y']

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Конвертация данных в тензоры PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Определение архитектуры нейронной сети
class StartupPredictor(nn.Module):
    def __init__(self, input_size):
        super(StartupPredictor, self).__init__()
        # Входной слой -> скрытый слой
        self.fc1 = nn.Linear(input_size, 64)  # 64 нейрона в скрытом слое
        # Скрытый слой -> выходной слой
        self.fc2 = nn.Linear(64, 1)  # Один выход (предсказание прибыли)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Применение функции активации ReLU
        x = self.fc2(x)  # Прямое предсказание
        return x

# Инициализация модели
input_size = X_train.shape[1]
model = StartupPredictor(input_size)

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()  # Среднеквадратичная ошибка
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
epochs = 1000
for epoch in range(epochs):
    # Обнуление градиентов
    optimizer.zero_grad()

    # Прямой проход (предсказание)
    outputs = model(X_train_tensor)

    # Вычисление ошибки
    loss = criterion(outputs, y_train_tensor)

    # Обратное распространение ошибки
    loss.backward()

    # Обновление весов
    optimizer.step()

    # Печать информации о потере каждые 100 эпох
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Тестирование модели на тестовых данных
model.eval()
with torch.no_grad():
    predicted = model(X_test_tensor)
    test_loss = criterion(predicted, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
```

### Что мы сделали в коде:

1. **Модель:**
   - Мы создали нейронную сеть с двумя слоями: входной (с 64 нейронами) и выходной (1 нейрон для предсказания).
   - В качестве функции активации мы использовали ReLU.

2. **Обучение:**
   - Мы использовали функцию потерь (среднеквадратичная ошибка) для вычисления разницы между предсказанием и реальными значениями.
   - Оптимизатор Adam обновляет веса сети для минимизации ошибки.

3. **Тестирование:**
   - После обучения мы оцениваем качество модели на тестовых данных.


## Работа с несколькими целевыми переменными

Когда у нас несколько целевых переменных, например, мы хотим предсказать одновременно прибыль через 1 год и количество инвесторов, архитектура модели будет немного отличаться. В случае нейронной сети мы можем просто изменить последний слой, чтобы он выводил несколько значений.

```
# Предположим, что целевые переменные - 'Projected_Profit_1Y' и 'Number_of_Investors'
y = data[['Projected_Profit_1Y', 'Number_of_Investors']]

# Изменение архитектуры модели
class StartupPredictorMultiTask(nn.Module):
    def __init__(self, input_size):
        super(StartupPredictorMultiTask, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Входной -> скрытый слой
        self.fc2 = nn.Linear(64, 2)  # 2 выхода (для двух целевых переменных)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Два выхода (прибыль и число инвесторов)
        return x

# Целевая переменная теперь содержит 2 столбца
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Определение функции потерь (среднеквадратичная ошибка для многозадачности)
criterion = nn.MSELoss()

# Обучение модели остается схожим, но теперь выходов больше
```

## Архитектура LSTM

LSTM отлично справляется с последовательными данными, например, если данные о стартапах зависят от прошлых событий или временных показателей. Она может запоминать информацию о последовательностях и делать более точные предсказания в таких задачах.

```
import torch
import torch.nn as nn
import torch.optim as optim

# Предположим, что у нас есть временные данные (например, за несколько лет по стартапам)
# Данные имеют размерность (количество примеров, длина последовательности, количество признаков)
# Для простоты возьмем данные без временной оси, но можно использовать временные ряды.

class StartupPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StartupPredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM возвращает все скрытые состояния и последнее скрытое состояние
        out, (hn, cn) = self.lstm(x)
        # Используем последнее скрытое состояние для предсказания
        out = self.fc(hn[-1])  # hn[-1] - это последнее скрытое состояние последнего слоя
        return out

# Параметры модели
input_size = X_train.shape[2]  # Количество признаков
hidden_size = 64  # Размер скрытого слоя LSTM
num_layers = 2  # Количество LSTM-слоев
output_size = 2  # Две целевые переменные

# Инициализация модели
model = StartupPredictorLSTM(input_size, hidden_size, num_layers, output_size)

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Входные данные для LSTM должны быть трехмерными: (batch_size, sequence_length, num_features)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, X_train.shape[1])
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, X_test.shape[1])
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 2)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 2)

# Обучение модели
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Тестирование модели
model.eval()
with torch.no_grad():
    predicted = model(X_test_tensor)
    test_loss = criterion(predicted, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
```

### Что здесь происходит:

#### LSTM:
В LSTM мы используем скрытые состояния для запоминания информации о предыдущих временных шагах. Выходом LSTM является скрытое состояние на каждом шаге последовательности. Мы используем последнее скрытое состояние для предсказания.

#### Многозадачное обучение:
В примере с несколькими целевыми переменными выходной слой модели предсказывает две переменные одновременно.

#### Когда использовать LSTM:
LSTM полезна, когда данные имеют временные зависимости или последовательности. Например, если успех стартапа зависит от прошлых финансовых показателей или трендов.

#### Следующие шаги:
- Можно экспериментировать с разными архитектурами LSTM (увеличивать/уменьшать количество слоев, нейронов).
- Попробовать другие рекуррентные архитектуры, такие как GRU.
