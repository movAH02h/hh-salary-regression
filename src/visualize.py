import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_results(y_test: np.ndarray, y_pred: np.ndarray, output_dir: str) -> None:
    """
    Визуализирует предсказания модели и сохраняет график.

    Аргументы:
        y_test (np.ndarray): Реальные значения.
        y_pred (np.ndarray): Предсказанные значения.
        output_dir (str): Папка для сохранения графика.

    Вернет:
        None
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Фактическая ЗП')
    plt.ylabel('Предсказанная ЗП')
    plt.title('Результаты регрессии: Факт vs Предсказание')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, 'regression_results.png'))
    plt.close()
    print(f"График сохранен в {output_dir}")
