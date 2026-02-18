import os
import joblib
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

def train_and_save_model(x_data: np.ndarray, y_data: np.ndarray, model_dir: str) -> str:
    """
    Обучает модель RandomForestRegressor и сохраняет её веса в файл.

    Аргументы:
        x_data (np.ndarray): Матрица признаков.
        y_data (np.ndarray): Вектор целевой переменной (ЗП).
        model_dir (str): Директория для сохранения модели.

    Вернет:
        str: Полный путь к сохраненному файлу весов.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_file = "model_weights.pkl"
    model_path = os.path.join(model_dir, model_file)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=5,
        max_features='sqrt'        
    )
    
    logger.info("Начинаю обучение модели регрессии...")
    model.fit(x_data, y_data)

    joblib.dump(model, model_path)
    logger.info(f"Модель успешно обучена и сохранена в {model_file}")
    
    return model_path
