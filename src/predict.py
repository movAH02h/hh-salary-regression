import os
import joblib
import numpy as np
import logging
from typing import List, Union, Any

logger = logging.getLogger(__name__)

def mock_app(x_path: str, model_dir: str, model_path: str) -> Union[List[float], str]:
    """
    Имитирует работу приложения для предсказания на основе .npy файла.

    Args:
        x_path (str): Путь к файлу .npy с данными.
        model_dir (str): Директория модели (для вывода ошибки).
        model_path (str): Путь к сохраненным весам .pkl.

    Returns:
        Union[List[float], str]: Список предсказаний или сообщение об ошибке.
    """
    try:
        loaded_model: Any = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Не удалось загрузить модель: {e}")
        return f"Ошибка: Сначала обучите модель и сохраните веса в {model_dir}"

    if not os.path.exists(x_path):
        return f"Ошибка: Файл {x_path} не найден"

    try:
        x_input: np.ndarray = np.load(x_path, allow_pickle=True)
        predictions: np.ndarray = loaded_model.predict(x_input)
        return [float(p) for p in predictions]
    except Exception as e:
        return f"Ошибка при обработке данных: {e}"
