"""
model.py — Обучение модели и сохранение весов в data/model_weights.mw

Запуск:
    python model.py

После выполнения в папке data/ появится файл model_weights.mw
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "gym_churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model_weights.mw")

# Загрузка данных
df = pd.read_csv(DATA_PATH)

# Удаляем признак Phone (низкая корреляция с таргетом, как выявлено в EDA)
df = df.drop("Phone", axis=1)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Разбивка и масштабирование
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Обучение модели
# SVC с линейным ядром — лучший результат по GridSearch в оригинальном ноутбуке
model = CalibratedClassifierCV(SVC(C=1.5, kernel="linear"), ensemble=False)
model.fit(X_train_sc, y_train)

# Метрики
train_acc = accuracy_score(y_train, model.predict(X_train_sc))
test_acc = accuracy_score(y_test,  model.predict(X_test_sc))

print(f"Точность на тренировочных данных : {train_acc:.4f}")
print(f"Точность на тестовых данных : {test_acc:.4f}")

# Сохранение весов
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump({"model": model, "scaler": scaler, "feature_names": list(X.columns)}, MODEL_PATH)
print(f"\nВеса модели сохранены в {MODEL_PATH}")
