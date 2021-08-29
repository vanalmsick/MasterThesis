import os

import mlflow.keras
import numpy as np
import pandas as pd

# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')

from app.z_helpers import helpers as my_helpers
from app.d_prediction.a_tf_base import median_scaling, _get_prep_data, _reformat_DF
from app.d_prediction.NN_tensorflow_models import TF_ERROR_METRICS

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from xgboost import XGBRegressor
from hyperopt.early_stop import no_progress_loss
import mlflow


def _optimize_obj(params):
    #mlflow.autolog()

    def _eval(model, X, y):
        y_pred = model.predict(X)
        print(y_pred)
        scores = {}
        for metric in  TF_ERROR_METRICS:
            scores[metric.name] = np.float(metric(y, y_pred))

        return scores


    dataset = _get_prep_data(params['train'], params['val'], params['test'], flatten=True, keep_last_n_periods=params['backlooking_period'])


    model = XGBRegressor(n_estimators=int(params['n_estimators']),
                         booster=params['booster'],
                         gamma=params['gamma'],
                         max_depth=params['max_depth'],
                         eta=params['eta'],
                         min_child_weight=params['min_child_weight'],
                         nthread=params['nthread'],
                         random_state=params['seed'],
                         verbosity=params['silent'],
                         subsample=params['subsample'],
                         colsample_bytree=params['colsample_bytree'],
                         tree_method=params['tree_method'])

    results = model.fit(dataset['train_X'].numpy(), dataset['train_y'].numpy(),
                        eval_set=[(dataset['train_X'].numpy(), dataset['train_y'].numpy()),
                                  (dataset['val_X'].numpy(), dataset['val_y'].numpy())],
                        eval_metric=params['eval_metric'],
                        verbose=params['silent'])

    final_results = {}
    final_results['train'] = _eval(results, dataset['train_X'].numpy(), dataset['train_y'].numpy())
    final_results['val'] = _eval(results, dataset['val_X'].numpy(), dataset['val_y'].numpy())
    final_results['test'] = _eval(results, dataset['test_X'].numpy(), dataset['test_y'].numpy())

    out = {'loss': final_results['val']['mean_absolute_error'], 'status': STATUS_OK, 'results': final_results, 'params': params}

    with mlflow.start_run() as run:
        mlflow.xgboost.log_model(results, '')
    mlflow_params = {'kwargs': params, 'model_name': 'XGBoost'}
    for set in ['train', 'val', 'test']:
        mlflow_params[f'metrics_{set}'] = final_results[set]
    mlflow_saved = my_helpers.mlflow_last_run_add_param(param_dict=mlflow_params)

    return out





def _find_optimal_model(train_ds, val_ds, test_ds, data_props, examples):

    search_space = {
        'backlooking_period': hp.choice('backlooking_period', [1, 2, 3, 4]),
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'mae',
        # Increase this number if you have more cores. Otherwise, remove it and it will default
        # to the maxium number.
        'nthread': None,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 0,
        'seed': 42
    }



    search_space['train'] = train_ds
    search_space['val'] = val_ds
    search_space['test'] = test_ds
    search_space['iter_step'] = data_props['iter_step']

    trials = Trials()
    best = fmin(_optimize_obj,
                search_space,
                algo=tpe.suggest,
                trials=trials,
                early_stop_fn=no_progress_loss(iteration_stop_count=25, percent_increase=0.025),
                max_evals=100)

    best_result = trials.best_trial['result']['results']
    best_params = trials.best_trial['result']['params']

    best_result = pd.DataFrame(best_result)
    best_result = _reformat_DF(best_result, data_props['iter_step'])
    best_params = pd.Series(best_params, name=data_props['iter_step'])

    return best_result, best_params






def run_model_acorss_time(data_obj, model_name, y_col, max_serach_iterations=200, redo_serach_best_model=False, max_backlooking=None, example_len=5, example_list=[], export_results=False):

    results_storage = {}

    def _reformat_DF(df, head):
        df = df.append(pd.Series(df.columns.tolist(), name='columns', index=df.columns.tolist()))
        df = df.append(pd.Series([head] * len(df.columns.tolist()), name='time_step', index=df.columns.tolist()))
        df.columns = [f'{head}_{col}' for col in df.columns.tolist()]
        return df

    results_storage = {'error': pd.DataFrame(), 'model': pd.DataFrame()}

    for out in [data_obj['200000_201500'], data_obj['200100_201600'], data_obj['200200_201700'], data_obj['200300_201800'], data_obj['200400_201900'], data_obj['200500_202000']]:
        print('Time-step:', out['iter_step'])
        train_ds, val_ds, test_ds = data_obj.tsds_dataset(out='all', out_dict=out)
        train_ds, val_ds, test_ds = median_scaling(train_ds, val_ds, test_ds, y_col_idx=out['columns_lookup']['X'][y_col])
        examples = data_obj.get_examples(example_len=example_len, example_list=example_list, y_col=y_col)
        examples['pred'] = {}
        data_props = data_obj.get_data_props()


        best_result, best_params = _find_optimal_model(train_ds, val_ds, test_ds, data_props, examples)

        results_storage['error'] = pd.concat([results_storage['error'].sort_index(), best_result.sort_index()], axis=1)
        results_storage['model'] = pd.concat([results_storage['model'].sort_index(), best_params.sort_index()], axis=1)

        if export_results != False:
            results_storage['error'].to_csv(export_results + f'results_{model_name}_error.csv')
            results_storage['model'].to_csv(export_results + f'results_{model_name}_model.csv')









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

    model_name = 'XGBoost'


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

    data.filter_y(just_include=y_pred_col)


    ############# RUN ALL MODELS ACROSS TIME #############
    mlflow.set_experiment(model_name)

    run_model_acorss_time(data, model_name, y_pred_col[0],export_results=export_results)

    ######################################################





