import pandas as pd
import numpy as np
import warnings
from src.preprocessing import preproc
from catboost import Pool
from src.aggregate import new_features

warnings.filterwarnings("ignore")


def calc_inference(data):

    cols = pd.read_pickle("cols.pkl")
    data = data[cols]
    data['date_report'] = data['ДатаНачалаЗадачи']
    # data['marker'] = 1
    data.sort_values("date_report", inplace=True)

    filename_attr = "data_mgz_attributes__24062023__1000_GMT3.csv"
    attr = pd.read_csv(filename_attr, sep=';', index_col='Unnamed: 0')

    data = preproc(data, attr, task_name="INFER")
    data = new_features(data)

    final_feat = pd.read_pickle("artifacts/final_feat.pkl")
    new_cat_feat = pd.read_pickle("artifacts/new_cat_feat.pkl")

    infer_pool = Pool(data=data[final_feat], cat_features=new_cat_feat)

    models = pd.read_pickle("artifacts/models.pkl")

    pred = 0

    for model in models:
        pred += model.predict(infer_pool)

    final_pred = np.round((pred / 4).clip(0))

    return final_pred