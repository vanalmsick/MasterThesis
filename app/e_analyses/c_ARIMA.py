import os

import numpy as np
import pandas as pd

# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')

from app.z_helpers import helpers as my_helpers

from pmdarima.arima import auto_arima
import mlflow
from statsmodels.tsa.arima.model import ARIMA
import statistics





def _find_optimal_model(train, val, test, data_props, examples):

    results = []
    for row in train['y_data']:
        model = auto_arima(row, trace=True)
        results.append(model.order)

    ARIMA_potentials = list(dict.fromkeys(results))

    val_results = {}
    for props in ARIMA_potentials:
        if props not in val_results:
            val_results[props] = {'val': {}, 'test': {}}
        for set, X, y in zip(['val', 'test'], [train['y_data'], np.concatenate((train['y_data'], val['y_data']), axis=1)], [val['y_data'], test['y_data']]):
            for i in range(len(X)):
                mod = ARIMA(X[i], order=props).fit()
                y_pred = float(mod.forecast())
                y_true = float(y[i])
                mae = abs(y_pred - y_true)
                mda = int(np.sign(y_pred) == np.sign(y_true) or np.sign(np.round(y_pred, 4)) == np.sign(np.round(y_true, 4)))
                mse = (y_pred - y_true) ** 2
                pos = int(np.sign(np.round(y_true, 6)))
                for err, vale in zip(['mae', 'mda', 'mse', 'pos'], [mae, mda, mse, pos]):
                    if 'mae' not in val_results[props][set]:
                        val_results[props][set] = {'mae': [], 'mda': [], 'mse': [], 'pos': []}
                    val_results[props][set][err].append(vale)

    final_results = {}
    for props in ARIMA_potentials:
        if props not in final_results.items():
            final_results[props] = {}
        for set in ['val', 'test']:
            for err, vals in val_results[props][set].items():
                final_results[props][f'{set}_{err}'] = statistics.mean(vals)
                if f'{set}_best_score' not in final_results or final_results[f'{set}_best_score'] > final_results[props][f'{set}_{err}']:
                    final_results[f'{set}_best_score'] = final_results[props][f'{set}_{err}']
                    final_results[f'{set}_best_param'] = props

    return final_results









def run_model_acorss_time(data_obj, model_name, y_col, max_serach_iterations=200, redo_serach_best_model=False, max_backlooking=None, example_len=5, example_list=[], export_results=False):

    results_storage = {}

    def _reformat_DF(df, head):
        df = df.append(pd.Series(df.columns.tolist(), name='columns', index=df.columns.tolist()))
        df = df.append(pd.Series([head] * len(df.columns.tolist()), name='time_step', index=df.columns.tolist()))
        df.columns = [f'{head}_{col}' for col in df.columns.tolist()]
        return df

    results_storage = {'model': pd.DataFrame(), 'search': pd.DataFrame()}

    for out in [data_obj['200000_201500'], data_obj['200100_201600'], data_obj['200200_201700'], data_obj['200300_201800'], data_obj['200400_201900'], data_obj['200500_202000']]:
        print('Time-step:', out['iter_step'])
        dataset = data.y_dataset(out='all', out_dict=out)
        examples = data_obj.get_examples(example_len=example_len, example_list=example_list, y_col=y_col)
        examples['pred'] = {}
        data_props = data_obj.get_data_props()


        best_result = _find_optimal_model(dataset['train'], dataset['val'], dataset['test'], data_props, examples)
        print(best_result)
        part_result = best_result[best_result['val_best_param']]
        part_result['params'] = best_result['val_best_param']
        part_result = pd.Series(part_result, name=out['iter_step'])

        for key in ['val_best_score', 'val_best_param', 'test_best_score', 'test_best_param']:
            best_result.pop(key)
        serach_log = pd.DataFrame(best_result)
        serach_log = _reformat_DF(serach_log, out['iter_step'])


        results_storage['model'] = pd.concat([results_storage['model'].sort_index(), part_result.sort_index()], axis=1)
        results_storage['search'] = pd.concat([results_storage['search'].sort_index(), serach_log.sort_index()], axis=1)

        if export_results != False:
            results_storage['model'].to_csv(export_results + f'results_{model_name}_model.csv')
            results_storage['search'].to_csv(export_results + f'results_{model_name}_search.csv')









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

    model_name = 'ARIMA'


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
    out = data['200000_201500']

    data.y_dataset(out_dict=out)

    ############# RUN ALL MODELS ACROSS TIME #############
    #mlflow.set_experiment(model_name)

    run_model_acorss_time(data, model_name, y_pred_col[0],export_results=export_results)

    ######################################################





