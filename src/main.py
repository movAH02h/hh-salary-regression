import os
import logging
import numpy as np
import joblib
from train import train_and_save_model
from predict import mock_app
from visualize import visualize_results

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    """Запускает основной процесс обучения, предсказания и визуализации."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    
    x_path = os.path.join(DATA_DIR, "x_data.npy")
    y_path = os.path.join(DATA_DIR, "y_data.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        logger.error("Файлы данных не найдены в папке data/. Запустите Проект №1.")
        return

    x_data = np.load(x_path, allow_pickle=True)
    y_data = np.load(y_path, allow_pickle=True)

    model_path = train_and_save_model(x_data, y_data, MODEL_DIR)

    predictions_list = mock_app(x_path, MODEL_DIR, model_path)
    if isinstance(predictions_list, list):
        logger.info(f"Получено {len(predictions_list)} предсказаний.")

    model = joblib.load(model_path)
    full_preds = model.predict(x_data)
    visualize_results(y_data, full_preds)

if __name__ == "__main__":
    main()
