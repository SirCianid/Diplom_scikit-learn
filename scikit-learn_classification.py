"""1. Установка необходимых библиотек, и загрузка датасета."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure, ylabel, plot, legend, grid, show, xlabel, title

"""2. Задаем переменную для датасета."""
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

"""3. Получение данных и целевой переменной. x - данные; y - целевая переменная, соответсвенно."""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""4. Масштабирование данных. Хотя логистическая регрессия и не так чувствительна к не масштабированым данным,
но все равно способна выдавать лучший результат если признаки масштабированы."""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""5. Настройка гиперпараметров, и обучение модели логистической регрессии. 
Этот метод идеально подходит в данной ситуации, т.к. выбранный датасет является задачей бинарной классификации."""
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['liblinear', 'saga', 'lbfgs'],
    'class_weight': [None, 'balanced']
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(LogisticRegression(random_state=42, max_iter=8000), param_grid, cv=skf, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("Лучшие параметры:", grid_search.best_params_)

"""6. Предсказываем результаты на тестовых данных."""
y_pred = best_model.predict(X_test_scaled)

"""7. Оцениваем качество модели"""
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

"""8. Визуализация данных с помощью ROC-кривой."""
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

figure(figsize=(8, 6))
plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
xlabel('False Positive Rate (FPR)')
ylabel('True Positive Rate (TPR)')
title('ROC-Кривая')
legend(loc="lower right")
grid(True)
show()
