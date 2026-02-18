import numpy as np
import logging
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_processed_data(data_dir):
    """
      Загружает npy файлы, созданные в первом проекте.

      Аргументы:
        data_dir (str): директория с файлами.

      Вернет:
        Tuple[np.ndarray, np.ndarray]: Кортеж из (X_data, y_data)
    """
    x_path = os.path.join(data_dir, 'x_data.npy')
    y_path = os.path.join(data_dir, 'y_data.npy')
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Массивы данных не найдены в папке data. Запустите сначала Проект №1.")
    
    return np.load(x_path), np.load(y_path)

def train_regression_model(X, y):
    """
      Обучает RandomForestRegressor с учетом замечаний по коду.

      Арументы:
        X (np.ndarray): Матрица признаков.
        y (np.ndarray): Вектор целевой переменной.

      Вернет:
        Tuple[RandomForestRegressor, np.ndarray, np.ndarray, np.ndarray]:
          обученная модель, X_test, y_test и предсказания y_pred.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=5,
        max_features='sqrt'
    )
    
    logger.info("Начинаю обучение модели регрессии...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Результаты: MAE = {mae:.2f}, R2 = {r2:.2f}")
    
    return model, X_test, y_test, y_pred
