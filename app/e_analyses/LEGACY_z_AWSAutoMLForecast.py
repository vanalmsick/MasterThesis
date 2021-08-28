import os, sys, time, shutil, warnings
import mlflow
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
#import eli5
#from eli5.sklearn import PermutationImportance

if __name__ == '__main__':  # must be in if condition because I am pusing parallel processing
    # Working directory must be the higher .../app folder
    from app.z_helpers import helpers as my
    my.convenience_settings()

    ### run tensorboard
    # MLFLOW_TRACKING_URI = 'file:' + '/Users/vanalmsick/Workspace/MasterThesis/cache/MLflow'  # with local tracking serveer
    MLFLOW_TRACKING_URI = 'http://0.0.0.0:5000'  # with remote tracking server with registry
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)




    # Dataset to use
    dataset_name = 'handpicked_dataset'
    yearly_data = True

    from app.b_data_cleaning import get_dataset_registry

    dataset_props = get_dataset_registry()[dataset_name]
    comp_col = dataset_props['company_col']
    time_cols = dataset_props['iter_cols']
    industry_col = dataset_props['industry_col']

    recache_raw_data = False
    redo_data_cleaning = False

    # Cleaining settings
    required_filled_cols_before_filling = ['sales', 'roe', 'ebit']
    required_filled_cols_after_filling = []
    drop_threshold_row_pct = 0.4
    drop_threshold_row_quantile = 0.15
    drop_threshold_col_pct = 0
    append_data_quality_col = False

    from app.c_data_prep.i_feature_engineering import get_clean_data, feature_engerneeing

    df_cleaned = get_clean_data(dataset_name, recache_raw_data=recache_raw_data, redo_data_cleaning=redo_data_cleaning,
                            comp_col=comp_col, time_cols=time_cols, industry_col=industry_col,
                            required_filled_cols_before_filling=required_filled_cols_before_filling,
                            required_filled_cols_after_filling=required_filled_cols_after_filling,
                            drop_threshold_row_pct=drop_threshold_row_pct,
                            drop_threshold_row_quantile=drop_threshold_row_quantile,
                            drop_threshold_col_pct=drop_threshold_col_pct,
                            append_data_quality_col=append_data_quality_col)
    print('Data cleaned:', len(df_cleaned))

    features = ['lev_thi']

    df_to_use = feature_engerneeing(dataset=df_cleaned, comp_col=comp_col, time_cols=time_cols,
                                industry_col=industry_col, yearly_data=yearly_data, all_features=features)
    print('Data to use:', len(df_to_use))

    #y_cols = ['y_eps', 'y_eps pct', 'y_dividendyield', 'y_dividendyield pct', 'y_dividendperstock', 'y_dividendperstock pct', 'y_roe', 'y_roe pct', 'y_roa', 'y_roa pct', 'y_EBIT', 'y_EBIT pct', 'y_Net income', 'y_Net income pct']
    df_to_use = df_to_use[abs(df_to_use['y_eps pct']) < 200]
    print('200% brder:', len(df_to_use))

    y_col = ['y_eps pct']
    # all_cols = [comp_col] + time_cols + y_col + ['1_Inventory', '3_CAPX', '4_RnD', '5_Gross Margin', '6_Sales & Admin. Exp.', '9_Order Backlog', '10_Labor Force', '11_FIFO/LIFO dummy']
    #
    # df_to_use = df_to_use[all_cols]
    #
    # df_to_use = df_to_use[(df_to_use[all_cols[4:-1]] < df_to_use[all_cols[4:-1]].quantile(.975)).all(axis=1)]
    # df_to_use = df_to_use[(df_to_use[all_cols[4:-1]] > df_to_use[all_cols[4:-1]].quantile(.025)).all(axis=1)]
    print('90-quantil brder:', len(df_to_use))


    from app.c_data_prep.ii_data_prep import data_prep
    data = data_prep(dataset=df_to_use, y_cols=y_col, iter_cols=time_cols, comp_col=comp_col, keep_raw_cols=[], drop_cols=[])

    qrt_multiplier = 1 if yearly_data else 4

    data.window(input_width=4 * qrt_multiplier, pred_width=qrt_multiplier, shift=1)

    # data.block_static_split(val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0,
                             train_time_steps=5 * qrt_multiplier * 2,
                             val_time_steps=qrt_multiplier * 2,
                             test_time_steps=qrt_multiplier * 2, shuffle=True)
    # data.single_time_rolling(val_time_steps=1, test_time_steps=1, shuffle=True)

    data.normalize(method='no')
    #data.normalize(method='block')
    # data.normalize(method='time')
    # data.normalize(method='set')

    data.compute()

    print(data)

    y_col = ['y_eps pct']
    #X_cols = y_col + ['1_Inventory', '3_CAPX', '4_RnD', '5_Gross Margin', '6_Sales & Admin. Exp.', '9_Order Backlog', '10_Labor Force', '11_FIFO/LIFO dummy']
    #data.filter_features(just_include=X_cols)
    data.filter_y(just_include=y_col)

    def key_value_table(df, keys=['ric', 'data_year', 'data_qrt'], file='AWS_AutoML_import.csv'):
        import hashlib
        new_data =[]
        cols = [col for col in df.columns.tolist() if col not in keys]
        with open(file, 'a') as f:
            f.write('metric_name,timestamp,metric_value,ric\n')
            for idx, row in df.iterrows():
                for col in cols:
                    comp = row[keys[0]]
                    comp = str(int(hashlib.sha1(comp.encode("utf-8")).hexdigest(), 16) % (10 ** 8))
                    print([comp] + [int(row[keys[1]])] + [col] + [row[col]])
                    f.write((str([col])[1:-1] + ',' + str(int(row[keys[1]])) + '-01-01,' + str(row[col]) + ',' + str(comp)).replace(' ','') + '\n')
        return new_data


    key_value_table(df_to_use[['ric', 'data_year', 'data_qrt'] + y_col], file='AWS_target_timeseries.csv')
    key_value_table(df_to_use[[col for col in df_to_use.columns.tolist() if col not in y_col]], file='AWS_related_timeseries.csv')
    #key_value_table(df_to_use[['ric', 'data_year', 'data_qrt'] + ['industry']], file='AWS_related_metadata.csv')

