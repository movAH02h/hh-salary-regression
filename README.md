# HH Salary Regression

Проект по предсказанию уровня заработной платы на основе предобработанных данных из HeadHunter.

## Описание
Данный модуль является вторым этапом пайплайна. Он использует очищенные признаки (массивы NumPy) для обучения модели `RandomForestRegressor`.

## Особенности модели
- **Алгоритм**: Случайный лес (Random Forest).
- **Гиперпараметры**: Оптимизированы для предотвращения переобучения (`min_samples_leaf=5`, `max_features='sqrt'`).
- **Метрики**: MAE и Коэффициент детерминации ($R^2$).

## Запуск
1. Убедитесь, что в папке `data/` находятся файлы `x_data.npy` и `y_data.npy` (результат работы Проекта №1).
2. Установите зависимости:
   ```bash
   pip install scikit-learn numpy matplotlib seaborn joblib


Следующее ДЗ: https://github.com/movAH02h/hh-it-classification.git
