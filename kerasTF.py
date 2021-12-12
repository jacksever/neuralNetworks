import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Считываем нашу табличку с данными
users = pd.read_csv("data/users.csv")

# Выводим ее
print(users)

# Заменим строки в числа, ИИ работают с числами!
n = {'no': 0, 'yes': 1}
users['License'] = users['License'].map(n)

# Выводим таблицу
print(users)

# Разбиваем дата сет на признаки и результат
feature_cols = ['Age', 'License', 'Experience', 'Accidents', 'Traffic_Violation', 'Rating']
X = users[feature_cols]  # Features
y = users.Result  # Результирующий столбец

# Нейросети оптимально работают со значениями от -1 до 1 или от 0 до 1 (функция активации relu), так как в этих
# промежутках наиболее характерны функции активации Отскейлим значения
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X)
X_scale = min_max_scaler.transform(X)

# Выводим признаки отскейленные, проверяем
print(X_scale)

# Разбиваем дата сет на часть для обучения, часть для проверки
# 70% обучение и 30% для проверки
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.3, random_state=1)

# Создаем модель нейронной сети
model = Sequential([
    # Создаем модель нейронной сети
    # Входной слой 6 нейронов, сразу за ним первый скрытый в 12 нейронов
    Dense(12, activation='relu', input_dim=6),
    # Скрытый слой в 24 нейрона
    Dense(24, activation='relu'),
    # Скрытый слой в 48 нейрона
    Dense(48, activation='relu'),
    # Скрытый слой в 24 нейрона
    Dense(24, activation='relu'),
    # Скрытый слой в 12 нейрона
    Dense(12, activation='relu'),
    # Выходной слой в 1 нейрон, значение от 0 до 1, где ближе к 0 - нет, ближе к 1 - да
    Dense(1, activation='sigmoid')
    # Если на выходе больше 2 (к примеру 16) вариантов:
    # Dense(16,activation='softmax')
])

# Собираем модель нейронной сети
model.compile(optimizer='adam',  # метод адаптивной оценки моментов
              loss='binary_crossentropy',  # Выход бинарный
              metrics=['accuracy'])  # Просим вывести метрики, в частности точность

# Если несколько вариантов
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучаем 100 раз на данных, валидируем по 30% выборке
nn = model.fit(X_train, y_train,
               epochs=100,
               validation_data=(X_test, y_test))

# Предсказание
# Готовим строку
rows = pd.DataFrame([[31, 1, 6, 2, 0, 4.0]],
                    columns=['Age', 'License', 'Experience', 'Accidents', 'Traffic_Violation', 'Rating'], dtype=int)
# Обязательно скейлим!
rows_scale = min_max_scaler.transform(rows)
# Проводим предсказание
result = model.predict(rows_scale)
# Выводим результат предсказания
print(result)
