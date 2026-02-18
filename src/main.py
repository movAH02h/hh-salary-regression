import logging
import os
import joblib
from train import load_processed_data, train_regression_model
from visualize import plot_results

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    try:
        X, y = load_processed_data(DATA_DIR)
        
        model, X_test, y_test, y_pred = train_regression_model(X, y)
        
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        joblib.dump(model, os.path.join(MODELS_DIR, 'salary_regressor.pkl'))
        
        plot_results(y_test, y_pred, MODELS_DIR)
        
        logger.info("Проект регрессии успешно выполнен.")
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")

if __name__ == "__main__":
    main()
