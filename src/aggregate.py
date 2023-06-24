import pandas as pd


def new_features(data: pd.DataFrame):
    agg_func = ['sum', 'mean', 'min', 'max', 'count', 'std', 'median']

    gp_obj_workers_day = data.groupby(["obj_key", 'ДатаНачалаЗадачи_dayofweek']).agg(
        {'Кол-во рабочих': agg_func}).reset_index()

    gp_obj_workers_day.columns = ["obj_key", 'ДатаНачалаЗадачи_dayofweek', 'sum_workers_day', 'mean_workers_day',
                                  'min_workers_day', 'max_workers_day', 'count_workers_day',
                                  'std_workers_day', 'median_workers_day']

    gp_obj_workers_month = data.groupby(["obj_key", 'ДатаНачалаЗадачи_month']).agg(
        {'Кол-во рабочих': agg_func}).reset_index()

    gp_obj_workers_month.columns = ["obj_key", 'ДатаНачалаЗадачи_month', 'sum_workers_month', 'mean_workers_month',
                                    'min_workers_month', 'max_workers_month', 'count_workers_month',
                                    'std_workers_month', 'median_workers_month']

    gp_obj_workers = data.groupby(["obj_key"]).agg(
        {'Кол-во рабочих': agg_func}).reset_index()
    gp_obj_workers.columns = ["obj_key", 'sum_workers', 'mean_workers',
                              'min_workers', 'max_workers', 'count_workers',
                              'std_workers', 'median_workers']

    data_gp = pd.merge(left=data, left_on='obj_key',
                       right=gp_obj_workers, right_on='obj_key',
                       how="left")

    data_gp = pd.merge(left=data_gp, left_on=['obj_key', 'ДатаНачалаЗадачи_month'],
                       right=gp_obj_workers_month, right_on=['obj_key', 'ДатаНачалаЗадачи_month'],
                       how="left")

    data_gp = pd.merge(left=data_gp, left_on=['obj_key', 'ДатаНачалаЗадачи_dayofweek'],
                       right=gp_obj_workers_day, right_on=['obj_key', 'ДатаНачалаЗадачи_dayofweek'],
                       how="left")

    return data_gp
