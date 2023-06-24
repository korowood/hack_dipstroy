import optuna
import pandas as pd
from optuna.samplers import TPESampler
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import random
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

seed = 7575

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

# drop_cols = ["target", "№ п/п",
#              'ДатаНачалаЗадачи', 'ДатаОкончанияЗадачи', 'ДатаначалаБП0', 'ДатаокончанияБП0',
#              'date_report']
#
# cat_feat = ['obj_prg', 'obj_subprg', 'Кодзадачи', 'НазваниеЗадачи', 'obj_key', 'состояние площадки', 'Экспертиза', ]



def seed_everything(seed=7575):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def start_search(data, calc_feat, cat_feat):
    seed_everything()

    def objective(trial):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

        X = data[calc_feat]#.drop(drop_cols, axis=1)
        y = data['target']

        # Определение количества разбиений для кросс-валидации
        n_splits = 4

        # Создание генератора разбиений для временной кросс-валидации
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Список для хранения метрик качества модели на каждом разбиении
        mae_scores = []

        param = {
            'loss_function': 'RMSE',
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
            'max_bin': trial.suggest_int('max_bin', 200, 400),
            'rsm': trial.suggest_uniform('rsm', 0.3, 1.0),
            # 'subsample': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.06, 0.12),
            'n_estimators': trial.suggest_categorical('n_estimators', [300]),
            'max_depth': trial.suggest_int('depth', 4, 15),
            # 'random_state': trial.suggest_categorical('random_state', [7575]),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
            'od_type': 'Iter',
            'od_wait': 20
        }
        model = CatBoostRegressor(**param, verbose=0, random_state=7575)

        for fold, (train_index, val_index) in enumerate(tscv.split(X)):
            # Разделение данных на обучающий и тестовый наборы
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            train_pool = Pool(data=X_train, label=y_train, cat_features=cat_feat)
            val_pool = Pool(data=X_val, label=y_val, cat_features=cat_feat)
            # test_pool = Pool(data=X_test, label=y_test, cat_features=cat_feat)
            # Обучение модели
            model.fit(train_pool, eval_set=val_pool)

            # Прогнозирование на тестовом наборе
            y_pred = model.predict(val_pool)

            # Вычисление метрики качества (в данном случае среднеквадратическая ошибка)
            mae = mean_absolute_error(y_val, y_pred)

            # Добавление значения метрики в список
            mae_scores.append(mae)
            print(f"FOLD: {fold+1}, MAE: {mae}")
        mean_mae = np.mean(mae_scores)
        print(f"среднее значение по фолдам: {mean_mae}")
        return mean_mae

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=10, n_jobs=-1)
    print('Количество итераций:', len(study.trials))
    print('Лучшие параметры:', study.best_trial.params)
    params_cat = study.best_params

    return params_cat
