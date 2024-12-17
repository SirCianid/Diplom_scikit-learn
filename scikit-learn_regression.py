"""1. Установка необходимых библиотек, и загрузка датасета."""
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

"""2. Задаем переменную для датасета."""
diabetes = load_diabetes()

"""3. Получение данных и целевой переменной. x - данные; y - целевая переменная, соответсвенно."""
X = diabetes.data
y = diabetes.target

"""4. Разделяем данные на обучающую и тестовую выборки."""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""5. Масштабирование данных. Т.к. выбранный метод обучения чувствителен к масштабу признаков, 
нужно стандартизировать и нормализовать признаки."""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""6. Настраиваем гиперпараметры для повышения точности предсказания. 
 Обучение модели с помощью метода опорных векторов (Support Vector Regression).
 P.S. gamma актуально для rbf и poly."""
param_grid = {
    'kernel': ['rbf', 'linear', 'poly'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 1],
    'epsilon': [0.01, 0.1, 0.2]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(SVR(), param_grid, cv=kf, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("Лучшие параметры:", grid_search.best_params_)

"""8. Предсказание на тестовых данных."""
y_pred = best_model.predict(X_test_scaled)

"""9. Оценка качества модели. Поскольку SVR - это регрессионная модель, 
оценка точности будет отличаться от классификации."""
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R^2:", r2)

"""10. Визуализация результатов."""
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Истинные значения")
plt.ylabel("Предсказанные значения")
plt.title("Предсказанные vs. Истинные значения")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.grid(True)
plt.show()
