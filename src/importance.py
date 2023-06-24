import eli5
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from eli5.sklearn import PermutationImportance, explain_prediction
import eli5
import os
import numpy as np
import random
from catboost import Pool, CatBoostRegressor

drop_cols = ["target", "№ п/п",
             'ДатаНачалаЗадачи', 'ДатаОкончанияЗадачи', 'ДатаначалаБП0', 'ДатаокончанияБП0',
             'date_report']

cat_feat = ['obj_prg', 'obj_subprg', 'Кодзадачи', 'НазваниеЗадачи', 'obj_key', 'состояние площадки', 'Экспертиза', ]


def seed_everything(seed=7575):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def search_top_k_features(data: pd.DataFrame):
    seed_everything()
    oot_data = data[data['date_report'] >= '2023-05-10'].copy()
    train_data = data[data['date_report'] < '2023-05-10'].copy()

    X_train, y_train = train_data.drop(drop_cols, axis=1), train_data['target']
    X_val, y_val = oot_data.drop(drop_cols, axis=1), oot_data['target']

    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_feat)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_feat)

    model = CatBoostRegressor(verbose=200, od_type="Iter", od_wait=20, iterations=1000)
    model.fit(train_pool, eval_set=val_pool)

    fi = pd.DataFrame(model.feature_importances_, columns=["w"])

    fi['features'] = X_train.columns
    fi = fi.sort_values("w", ascending=False)
    print("Признаки отобраны ... ", end='\n')
    final_feat = fi['features'].tolist()[:20]
    new_cat_feat = [f for f in final_feat if f in cat_feat]
    print(f"TOP20 отобранных признаков {final_feat}", end='\n')

    return final_feat, new_cat_feat
