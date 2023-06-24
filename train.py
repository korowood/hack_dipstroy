import pandas as pd
from src.preprocessing import preproc
from src.calc_params import start_search
from src.importance import search_top_k_features
from src.modeling import calc_final_models
from src.aggregate import new_features
import time

filename_data = "dataset_hackaton_ksg__v2__23062023__1710_GMT3.csv"
filename_attr = "data_mgz_attributes__24062023__1000_GMT3.csv"

if __name__ == '__main__':
    start_time = time.time()
    print("Идет загрузка и подготовка данных ... ")
    df = pd.read_csv(filename_data, sep=';', index_col='Unnamed: 0')
    attr = pd.read_csv(filename_attr, sep=';', index_col='Unnamed: 0')

    data = preproc(df, attr)

    data = new_features(data)
    # отбор признаков
    print("Начало отбора признаков ... ")
    final_feat, new_cat_feat = search_top_k_features(data)

    pd.to_pickle(final_feat, "artifacts/final_feat.pkl")
    pd.to_pickle(new_cat_feat, "artifacts/new_cat_feat.pkl")

    # расчет гиперпараметров
    print("Старт подбора гиперпараметров на отобранных признаках ... ")
    params = start_search(data, final_feat, new_cat_feat)
    pd.to_pickle(params, "artifacts/params.pkl")

    print("Расчет финальной модели")
    models, y_pred = calc_final_models(data, final_feat, new_cat_feat, params)
    pd.to_pickle(models, "artifacts/models.pkl")
    end_time = time.time()
    print(f"Общее время расчета {(end_time - start_time) / 60 :.2f} min")