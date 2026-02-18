import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

def visualize_results(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Визуализирует результаты через графики рассеяния и гистограмму остатков.

    Аргументы:
        y_true (np.ndarray): Истинные значения.
        y_pred (np.ndarray): Предсказанные значения.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Размеры массивов y_true и y_pred не совпадают.")

    logger.info("Генерация графиков визуализации результатов...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, ax=axes[0])
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_title('Предсказанные значения vs Реальные')

    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, ax=axes[1], color='purple')
    axes[1].set_title('Распределение остатков (Ошибок)')

    plt.tight_layout()
    plt.show()

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logger.info(f"Средняя абсолютная ошибка (MAE): {mae:.2f} руб.")
    logger.info(f"Коэффициент детерминации (R2): {r2:.2f}")
