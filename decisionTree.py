import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Подключение классификатора дерева решений
from sklearn.model_selection import train_test_split  # Подключение функции для разделения выборки для обучения и теста
from sklearn import metrics  # Подключение метрик

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
# Считываем дата сет

users = pd.read_csv("data/users.csv")
print(users)

n = {'no': 0, 'yes': 1}
users['License'] = users['License'].map(n)

print(users)

# Разбиваем дата сет на признаки и результат
feature_cols = ['Age', 'License', 'Experience', 'Accidents', 'Traffic_Violation', 'Rating']
X = users[feature_cols]  # Features
y = users.Result  # Результирующий столбец

# Разбиваем дата сет
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  # 80% обучение и 20% тест

# Создаем классификатор дерева решения
clf = DecisionTreeClassifier()

# Тренируем дерево решения
clf = clf.fit(X_train, y_train)

# Предсказываем и тестируем на результат (сравнивая то что дает дерево с 20% сетом)
y_pred = clf.predict(X_test)

# Выводим отчет, на сколько наше дерево точно?
print("Точность:", metrics.accuracy_score(y_test, y_pred))

# Получаем картинку
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('users.png')
Image(graph.create_png())

row = pd.DataFrame([[26, 1, 3, 4, 0, 3.0]],
                   columns=['Age', 'License', 'Experience', 'Accidents', 'Traffic_Violation', 'Rating'],
                   dtype=float)
print(clf.predict(row))
