import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np


def calc_final_models(data: pd.DataFrame, final_feat, cat_feat, params):
    params['n_estimators'] = 600
    # Разделение на признаки и целевую переменную
    X = data[final_feat]
    y = data['target']

    # Определение количества разбиений для кросс-валидации
    n_splits = 4

    # Создание генератора разбиений для временной кросс-валидации
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Список для хранения метрик качества модели на каждом разбиении
    mae_scores = []

    models = []

    # Цикл по разбиениям временной кросс-валидации
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        # Разделение данных на обучающий и тестовый наборы
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_feat)
        val_pool = Pool(data=X_val, label=y_val, cat_features=cat_feat)
        # Обучение модели
        model = CatBoostRegressor(**params, verbose=0)
        model.fit(train_pool, eval_set=val_pool)

        # Прогнозирование на тестовом наборе
        y_pred = model.predict(val_pool)
        y_pred = y_pred.clip(0)

        # Вычисление метрики качества (в данном случае среднеквадратическая ошибка)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # Добавление значения метрики в список
        mae_scores.append(mae)
        models.append(model)
        print(f"FOLD {fold+1} | MAE: {mae:.2f} | R2: {r2:.2f}")

    # Вывод средней метрики качества по всем разбиениям
    print(f'Mean MAE: {np.mean(mae_scores):2f}')
    return models, y_pred
