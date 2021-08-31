import os

import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime

# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')
from app.z_helpers import helpers as my_helpers

from app.d_model_building.tensorflow_compiling import compile_and_fit, evaluate_model

class BaselineLastAvg(tf.keras.Model):
    def __init__(self, label_index, periods=1):
        super().__init__()
        self.label_index = label_index
        self.periods = periods

    def call(self, inputs):
        result = inputs[:, -self.periods:, self.label_index]
        result = result[:, :, tf.newaxis]
        final_result = tf.math.reduce_mean(result, axis=1)

        return final_result

    @property
    def name(self):
        return f'baseline_last{self.periods}_avg'


class BaselineStaticValue(tf.keras.Model):
    def __init__(self, value=0):
        super().__init__()
        self.value = value

    def call(self, inputs):
        result = tf.ones((1, 1, 1), dtype=tf.dtypes.float32)
        final_result = result * self.value

        return final_result

    @property
    def name(self):
        return f'baseline_static_{round(self.value,2)}'


def _collect_all_metrics(model, train_ds, val_ds, test_ds, mlflow_additional_params, data_props):
    train_performance = dict(zip(model.metrics_names, evaluate_model(model=model, tf_data=train_ds)))
    val_performance = dict(zip(model.metrics_names, evaluate_model(model=model, tf_data=val_ds)))
    test_performance = dict(zip(model.metrics_names, evaluate_model(model=model, tf_data=test_ds, mlflow_additional_params=mlflow_additional_params)))

    return pd.DataFrame({'train': train_performance, 'val': val_performance, 'test': test_performance})

from app.d_model_building.tensorflow_model_base import median_scaling



def run_model_acorss_time(data_obj, y_col, example_len=5, example_list=[], export_results=False, redo_serach_best_model=False):
    mlflow.set_experiment('baseline')
    experiment_date_time = int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    val_error_storage_all = pd.DataFrame()
    test_error_storage_all = pd.DataFrame()

    def _reformat_DF(df, head):
        df = df.append(pd.Series(df.columns.tolist(), name='columns', index=df.columns.tolist()))
        df = df.append(pd.Series([head] * len(df.columns.tolist()), name='time_step', index=df.columns.tolist()))
        df.columns = [f'{head}_{col}' for col in df.columns.tolist()]
        return df


    for out in [data_obj['200000_201500'], data_obj['200100_201600'], data_obj['200200_201700'], data_obj['200300_201800'], data_obj['200400_201900'], data_obj['200500_202000']]:
        print('Time-step:', out['iter_step'])

        val_error_storage = []
        test_error_storage = []

        train_ds, val_ds, test_ds = data_obj.tsds_dataset(out='all', out_dict=out)

        examples = data_obj.get_examples(example_len=example_len, example_list=example_list, y_col=y_col)
        examples['pred'] = {}
        data_props = data_obj.get_data_props()

        train_ds, val_ds, test_ds = median_scaling(train_ds, val_ds, test_ds, y_col_idx=out['columns_lookup']['X'][y_pred_col[0]])

        baseline1 = BaselineLastAvg(label_index=out['columns_lookup']['X'][y_pred_col[0]], periods=1)
        history, mlflow_additional_params = compile_and_fit(train=train_ds, val=val_ds, model=baseline1, MAX_EPOCHS=1, model_name=model_name)
        metrics = _collect_all_metrics(baseline1, train_ds, val_ds, test_ds, mlflow_additional_params, data_props)
        val_error_storage.append(metrics['val'].rename('baseline_last'))
        test_error_storage.append(metrics['test'].rename('baseline_last'))



        baseline2 = BaselineLastAvg(label_index=out['columns_lookup']['X'][y_pred_col[0]], periods=2)
        baseline3 = BaselineLastAvg(label_index=out['columns_lookup']['X'][y_pred_col[0]], periods=3)
        baseline4 = BaselineLastAvg(label_index=out['columns_lookup']['X'][y_pred_col[0]], periods=4)


        best_avg_loss = 10 ** 10
        best_avg_model = None
        best_avg_periods = 0
        for periods, model in zip([1, 2, 3, 4], [baseline1, baseline2, baseline3, baseline4]):
            history, mlflow_additional_params = compile_and_fit(train=train_ds, val=val_ds, model=model, MAX_EPOCHS=1, model_name='baseline_avg')
            val_performance = dict(zip(model.metrics_names, evaluate_model(model=model, tf_data=val_ds)))
            loss = val_performance['loss']
            if loss < best_avg_loss:
                best_avg_model = model
                best_avg_loss = loss
                best_avg_periods = periods
        metrics = _collect_all_metrics(best_avg_model, train_ds, val_ds, test_ds, mlflow_additional_params, data_props)
        val_error_storage.append(metrics['val'].rename(f'baseline_avg(t={best_avg_periods})'))
        test_error_storage.append(metrics['test'].rename(f'baseline_avg(t={best_avg_periods})'))
        print('######### Best Last Avg:', best_avg_periods, best_avg_loss)


        best_val_loss = 10 ** 10
        best_val_value = None
        for val in np.arange(-0.1, 0.1, 0.01):
            baseline0 = BaselineStaticValue(value=val)
            history, mlflow_additional_params = compile_and_fit(train=train_ds, val=val_ds, model=baseline0, MAX_EPOCHS=1, model_name=model_name)
            val_performance = dict(zip(baseline0.metrics_names, evaluate_model(model=baseline0, tf_data=val_ds)))
            loss = val_performance['loss']
            if loss < best_val_loss:
                best_val_value = val
                best_val_loss = loss
        print('######### Best Baseline Value:', best_val_value, best_val_loss)
        baseline0 = BaselineStaticValue(value=best_val_value)
        history, mlflow_additional_params = compile_and_fit(train=train_ds, val=val_ds, model=baseline0, MAX_EPOCHS=1, model_name=model_name)
        metrics = _collect_all_metrics(baseline0, train_ds, val_ds, test_ds, mlflow_additional_params, data_props)
        val_error_storage.append(metrics['val'].rename(f'baseline_static_value({round(best_val_value, 2)})'))
        test_error_storage.append(metrics['test'].rename(f'baseline_static_value({round(best_val_value, 2)})'))

        val_error = pd.DataFrame(val_error_storage).transpose()
        val_error = _reformat_DF(df=val_error, head=out['iter_step'])
        val_error_storage_all = pd.concat([val_error_storage_all.sort_index(), val_error.sort_index()], axis=1)

        test_error = pd.DataFrame(test_error_storage).transpose()
        test_error = _reformat_DF(df=test_error, head=out['iter_step'])
        test_error_storage_all = pd.concat([test_error_storage_all.sort_index(), test_error.sort_index()], axis=1)

        if export_results != False:
            val_error_storage_all.to_csv(export_results + f'results_baseline_models_val.csv')
            test_error_storage_all.to_csv(export_results + f'results_baseline_models_test.csv')







if __name__ == '__main__':
    import shutil, time

    # Working directory must be the higher .../app folder
    from app.z_helpers import helpers as my
    my.convenience_settings()

    ############################## USER SETTINGS ##############################

    # MLflow model registry location
    MLFLOW_TRACKING_URI = 'file:' + '/Users/vanalmsick/Workspace/MasterThesis/cache/MLflow'  # with local tracking serveer
    #MLFLOW_TRACKING_URI = 'http://0.0.0.0:5000'  # with remote tracking server with registry

    # dataset selection
    dataset_name = 'handpicked_dataset'
    yearly_data = True  # False = quarterly data

    # re-caching
    recache_raw_data = False
    redo_data_cleaning = False

    # data cleaning
    required_filled_cols_before_filling = ['sales', 'roe', 'ebit']
    required_filled_cols_after_filling = []
    drop_threshold_row_pct = 0.4
    drop_threshold_row_quantile = 0.15
    drop_threshold_col_pct = 0
    append_data_quality_col = False

    # feature engerneeing
    # features = ['lev_thi']  # just lev & thi columns
    features = 'all'  # all columns

    # y prediction column
    y_pred_col = ['y_eps pct']

    # window settings
    backlooking_yeras = 4

    # results location
    # export_results = False
    export_results = '/Users/vanalmsick/Workspace/MasterThesis/results/'

    model_name = 'dense_lit_best'

    ###########################################################################



    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    tracking_address = my_helpers.get_project_directories(key='tensorboard_logs')
    try:
        shutil.rmtree(tracking_address)
        time.sleep(10)
    except:
        pass
    os.mkdir(tracking_address)



    from app.b_data_cleaning import get_dataset_registry

    dataset_props = get_dataset_registry()[dataset_name]
    comp_col = dataset_props['company_col']
    time_cols = dataset_props['iter_cols']
    industry_col = dataset_props['industry_col']



    from app.c_data_prep.i_feature_engineering import get_clean_data, feature_engerneeing

    df_cleaned = get_clean_data(dataset_name, recache_raw_data=recache_raw_data, redo_data_cleaning=redo_data_cleaning,
                                comp_col=comp_col, time_cols=time_cols, industry_col=industry_col,
                                required_filled_cols_before_filling=required_filled_cols_before_filling,
                                required_filled_cols_after_filling=required_filled_cols_after_filling,
                                drop_threshold_row_pct=drop_threshold_row_pct,
                                drop_threshold_row_quantile=drop_threshold_row_quantile,
                                drop_threshold_col_pct=drop_threshold_col_pct,
                                append_data_quality_col=append_data_quality_col)



    df_to_use = feature_engerneeing(dataset=df_cleaned, comp_col=comp_col, time_cols=time_cols,
                                    industry_col=industry_col, yearly_data=yearly_data, all_features=features)



    y_cols = ['y_eps', 'y_eps pct', 'y_dividendyield', 'y_dividendyield pct', 'y_dividendperstock', 'y_dividendperstock pct', 'y_roe', 'y_roe pct', 'y_roa', 'y_roa pct', 'y_EBIT', 'y_EBIT pct', 'y_Net income', 'y_Net income pct']



    from app.c_data_prep.ii_data_prep import data_prep

    data = data_prep(dataset=df_to_use, y_cols=y_cols, iter_cols=time_cols, comp_col=comp_col)

    qrt_multiplier = 1 if yearly_data else 4

    data.window(input_width=backlooking_yeras * qrt_multiplier, pred_width=qrt_multiplier, shift=1)

    # data.block_static_split(val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=5 * qrt_multiplier * 2,
                             val_time_steps=qrt_multiplier,
                             test_time_steps=qrt_multiplier, shuffle=True)
    # data.single_time_rolling(val_time_steps=1, test_time_steps=1, shuffle=True)

    data.normalize(method='no')
    # data.normalize(method='block')
    #data.normalize(method='time')
    # data.normalize(method='set')

    data.compute()
    print(data)

    data.filter_features(just_include=['1_Inventory', '3_CAPX', '4_RnD', '5_Gross Margin', '6_Sales & Admin. Exp.', '9_Order Backlog', '10_Labor Force', '11_FIFO/LIFO dummy'] + y_pred_col)
    data.filter_y(just_include=y_pred_col)


    ############# RUN ALL MODELS ACROSS TIME #############


    run_model_acorss_time(data_obj=data, example_len=5, example_list=[], y_col=y_pred_col[0], export_results=export_results, redo_serach_best_model=False)


    ######################################################





