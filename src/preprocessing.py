import pandas as pd
import warnings

warnings.filterwarnings("ignore")

importance_list = ['1', '1.4.2', '3', '3.1', '3.3', '3.8', '3.11', '3.13', '3.15', '4',
                   '4.1.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8',
                   '4.9', '4.10', '4.11', '4.12', '5',
                   '5.11', '7.1', '7.1.8', '7.4', '8']


def create_sub_calendar_features(data: pd.DataFrame, col: str):
    data[col] = pd.to_datetime(data[col])
    data[f'{col}_week'] = data[col].dt.week
    data[f'{col}_day'] = data[col].dt.day
    data[f'{col}_dayofweek'] = data[col].dt.dayofweek
    data[f'{col}_month'] = data[col].dt.month
    return data


def preproc(df: pd.DataFrame, attr: pd.DataFrame, task_name="TRAIN"):
    # перевод даты в datetime
    df['ДатаОкончанияЗадачи'] = pd.to_datetime(df['ДатаОкончанияЗадачи'])
    df['ДатаНачалаЗадачи'] = pd.to_datetime(df['ДатаНачалаЗадачи'])
    df['date_report'] = pd.to_datetime(df['date_report'])
    attr['date_report'] = pd.to_datetime(attr['date_report'])

    # формирование таргета
    df['target'] = (df['ДатаОкончанияЗадачи'] - df['ДатаНачалаЗадачи']).dt.days
    # очистка датафрейма от пустых значений
    df = df[~df['target'].isna()]
    df['target'] = df['target'].astype(int)

    # приведение колонок атрибутов к нужному виду
    attr.rename(columns={"Код ДС": "obj_key"}, inplace=True)

    # обьеденение данных с атрибутами
    data = pd.merge_asof(df, attr, on=['date_report'], direction="forward", by='obj_key')

    # удаление выбросов
    if task_name == "TRAIN":
        data = data[data['target'] <= data['target'].mean() + data['target'].std()]

    # создание новых признаков из даты
    # TODO возможно поменять
    data = create_sub_calendar_features(data, 'ДатаНачалаЗадачи')
    data = create_sub_calendar_features(data, 'ДатаначалаБП0')
    data = create_sub_calendar_features(data, 'ДатаокончанияБП0')

    # замена пропуском для категориальных признаков
    data['НазваниеЗадачи'] = data['НазваниеЗадачи'].fillna(value="Uknown")
    data['Экспертиза'] = data['Экспертиза'].fillna(value="Uknown")
    data['состояние площадки'] = data['состояние площадки'].fillna(value="Uknown")
    data['flag'] = data['Кодзадачи'].apply(lambda x: 1 if x in importance_list else 0)
    return data
