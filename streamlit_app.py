import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
#from src.preprocessing import preproc
#from src.aggregate import new_features


PATH_TO_MODEL = 'artifacts/models.pkl'
PATH_TO_TEST_PREDICTIONS = 'artifacts/test_predictions.csv'
COLS_TO_DROP = [
    "target",
    "№ п/п",
    'ДатаНачалаЗадачи',
    'ДатаОкончанияЗадачи', 
    'ДатаначалаБП0',
    'ДатаокончанияБП0',
    'date_report'
]
OBJECT_KEY = 'Кодзадачи'
TARGET = 'target'
IMPORTANT_TASK_CODES = ()


@st.cache_resource
def load_model(path: str):
#    model = CatBoostRegressor()
#    model.load_model(path)
    models = pd.read_pickle(path)
    return models


@st.cache_data
def load_test_data():
    test = pd.read_csv(PATH_TO_TEST_PREDICTIONS)
    test = test.drop(['Кол-во дней'], axis=1)
    return test


@st.cache_data
def preprocess(uploaded_file):
    # Preprocessing code here
    features = pd.read_csv(uploaded_file)
#    cols = pd.read_pickle("cols.pkl")
#    
#    features = features[cols]
#    features['date_report'] = features['ДатаНачалаЗадачи']
#    # data['marker'] = 1
#    features.sort_values("date_report", inplace=True)
#    filename_attr = "data_mgz_attributes__24062023__1000_GMT3.csv"
#    attr = pd.read_csv(filename_attr, sep=';', index_col='Unnamed: 0')
#    data = preproc(features, attr, task_name="INFER")
#    data = new_features(data)
    target = features[TARGET]
    return features, target


@st.cache_data
def get_preds(_model, X):
    for model in models:
        pred += model.predict(infer_pool)

    final_pred = np.round((pred / 4).clip(0))
    return final_pred


#@st.cache_data
#def get_shap_values(_model, features):
#    explainer = shap.Explainer(_model)
#    shap_values = explainer(features)
#    return explainer, shap_values

@st.cache_data
def get_shap_values():
    shap_values = pd.read_pickle('artifacts/shap_values.pkl')
    return shap_values


@st.cache_data
def get_row_index(features: pd.DataFrame, value: str, key: str = OBJECT_KEY):
    assert OBJECT_KEY in features, (
        f'Feature columns must contain "{OBJECT_KEY}" column.'
    )
    features_filtered = features[features[key] == value]
    return features_filtered.index[0]


def choose_object(X, key: str = OBJECT_KEY):
    unique_keys = sorted(X[key].unique())
    object_value = st.selectbox('Выберите код задачи', unique_keys)
    return object_value


def main():
    uploaded_file = st.file_uploader("Выберите файл в формате .csv")
    if uploaded_file is not None:
        features, target = preprocess(uploaded_file)
#        models = load_model(PATH_TO_MODEL)
        test_cases = load_test_data()
#        preds = get_preds(models, features)
#        features['preds'] = np.abs(preds - target)
#        features_deduplicated = (
#            features[[OBJECT_KEY, 'preds']]
#            .drop_duplicates(subset=[OBJECT_KEY], keep='first')
#        )
        test_cases_with_preds = test_cases.merge(
            features,
            left_on=OBJECT_KEY,
            right_on=OBJECT_KEY,
            how='left',
        )
        test_cases_with_preds = test_cases_with_preds.rename(columns={'preds': 'Кол-во дней'})
        # Dataframe
        st.dataframe(test_cases_with_preds[['Кодзадачи', 'Название задачи', 'Кол-во дней']])

#        shap_values = get_shap_values(model, features)
        shap_values = get_shap_values()
        object_value = choose_object(features)
        object_index = get_row_index(features, object_value)
        
        # waterfall plot
        fig = plt.figure(figsize=(20, 20))
        max_display = max([np.count_nonzero(shap_values.values[i]) for i in range(len(features))])
        shap.plots.waterfall(shap_values[object_index], max_display=max_display, show=False)
        # force plot
        # shap.force_plot(explainer.expected_value, shap_values.values[object_index], show=False)
        plt.title("SHAP waterfall plot")
        plt.xlabel("SHAP value")
        st.pyplot(fig)


if __name__ == '__main__':
    main()
