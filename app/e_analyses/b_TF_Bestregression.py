import warnings
import datetime
import tensorflow as tf
import tensorflow_addons as tfa
import os
import mlflow.keras
import datetime
import pandas as pd

# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')
from app.z_helpers import helpers as my_helpers

from app.d_prediction.NN_tensorflow_models import compile_and_fit, evaluate_model
from app.d_prediction.prediction import plot


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

import math
def FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes):
    layers = []

    if n_layers == 1:
        layers.append(last_layer_nodes)
    else:
        nodes_increment = (last_layer_nodes - first_layer_nodes) / (n_layers - 1)
        nodes = first_layer_nodes
        for i in range(1, n_layers + 1):
            layers.append(math.ceil(nodes))
            nodes = nodes + nodes_increment

    return layers

def createmodel(n_layers, first_layer_nodes, last_layer_nodes, activation_func, input_size, output_size, **kwargs):
    model = Sequential()
    n_nodes = FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes)
    for i in range(1, n_layers):
        if i == 1:
            model.add(Dense(first_layer_nodes, input_dim=input_size, activation=activation_func))
        else:
            model.add(Dense(n_nodes[i - 1], activation=activation_func))

    # Finally, the output layer should have a single node in binary classification
    model.add(Dense(output_size, activation=activation_func))
    #model.compile(optimizer='adam', loss=loss_func, metrics=["accuracy"])  # note: metrics could also be 'mse'

    return model

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

def main_run_linear_models(train_ds, val_ds, test_ds, val_performance_dict, test_performance_dict, data_props, examples=None):

    train_X, train_y = tf.data.experimental.get_single_element(train_ds)
    train_y = tf.squeeze(train_y)
    val_X, val_y = tf.data.experimental.get_single_element(val_ds)
    val_y = tf.squeeze(val_y)
    test_X, test_y = tf.data.experimental.get_single_element(test_ds)
    test_y = tf.squeeze(test_y)

    periods = 1
    best_score = 100000000000000


    model = KerasRegressor(build_fn=createmodel) #, verbose=False
    activation_funcs = ['sigmoid', 'relu', 'tanh']
    param_grid = dict(n_layers=hp.choice('n_layers', [1, 2, 3]),
                      first_layer_nodes=hp.choice('first_layer_nodes',[128, 64, 32, 16]),
                      last_layer_nodes=hp.choice('last_layer_nodes',[32, 16, 4]),
                      activation_func=hp.choice('activation_func',activation_funcs),
                      input_size=train_X.shape[-1], output_size=1,
                      backlooking_window=hp.choice('backlooking_window', [1, 2, 3, 4]))

    warnings.filterwarnings('ignore')

    def _optimize_objective(kwargs):
        now = datetime.datetime.now()
        date_time = str(now.strftime("%y%m%d%H%M%S"))
        model_name = f"{date_time}_linear_{kwargs['backlooking_window']}_{kwargs['n_layers']}"
        periods = kwargs['backlooking_window']

        if periods == 'all':
            tmp_train_X = tf.reshape(train_X, (train_X.shape[0], -1))
            tmp_val_X = tf.reshape(val_X, (val_X.shape[0], -1))
            tmp_test_X = tf.reshape(test_X, (test_X.shape[0], -1))
        else:
            tmp_train_X = tf.reshape(train_X[:, -periods:, :], (train_X.shape[0], -1))
            tmp_val_X = tf.reshape(val_X[:, -periods:, :], (val_X.shape[0], -1))
            tmp_test_X = tf.reshape(test_X[:, -periods:, :], (test_X.shape[0], -1))
        tmp_train_ds = tf.data.Dataset.from_tensors((tmp_train_X, train_y))
        tmp_val_ds = tf.data.Dataset.from_tensors((tmp_val_X, val_y))
        tmp_test_ds = tf.data.Dataset.from_tensors((tmp_test_X, test_y))

        kwargs['input_size'] = tmp_train_X.shape[-1]

        model = createmodel(**kwargs)
        history, mlflow_additional_params = compile_and_fit(model=model, train=tmp_train_ds, val=tmp_val_ds,
                                                            MAX_EPOCHS=800, patience=25, model_name=model_name,
                                                            verbose=0)
        ML_kwargs = dict(kwargs)
        try:
            ML_kwargs.pop('batch_size')
        except:
            pass
        try:
            ML_kwargs.pop('epochs')
        except:
            pass
        mlflow_additional_params['kwargs'] = ML_kwargs

        val_performance_dict[model_name] = evaluate_model(model=model, tf_data=tmp_val_ds)
        test_performance_dict[model_name] = evaluate_model(model=model, tf_data=tmp_test_ds,
                                                           mlflow_additional_params=mlflow_additional_params)

        my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params)

        tf.keras.backend.clear_session()

        return_metrics = dict(loss=val_performance_dict[model_name][0],
                              status=STATUS_OK)

        return return_metrics

    trials = Trials()
    best = fmin(fn=_optimize_objective,
                space=param_grid,
                algo=tpe.suggest,
                max_evals=80,
                trials=trials)
    print('DONE\n', best)





if __name__ == '__main__':
    import shutil, time

    # Working directory must be the higher .../app folder
    from app.z_helpers import helpers as my
    my.convenience_settings()

    ### run tensorboard
    MLFLOW_TRACKING_URI = 'file:' + '/Users/vanalmsick/Workspace/MasterThesis/cache/MLflow'  # with local tracking serveer
    #MLFLOW_TRACKING_URI = 'http://0.0.0.0:5000'  # with remote tracking server with registry
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    tracking_address = my_helpers.get_project_directories(key='tensorboard_logs')
    try:
        shutil.rmtree(tracking_address)
        time.sleep(10)
    except:
        pass
    os.mkdir(tracking_address)

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

    features = ['lev_thi']
    features = 'all'
    df_to_use = feature_engerneeing(dataset=df_cleaned, comp_col=comp_col, time_cols=time_cols,
                                    industry_col=industry_col, yearly_data=yearly_data, all_features=features)

    from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

    #scaler_cols = [col for col in df_to_use.columns.tolist() if col not in time_cols and col != comp_col]
    #scaler = RobustScaler().fit(df_to_use[scaler_cols])
    #df_to_use[scaler_cols] = pd.DataFrame(scaler.transform(df_to_use[scaler_cols]), columns=scaler_cols)

    y_cols = ['y_eps', 'y_eps pct', 'y_dividendyield', 'y_dividendyield pct', 'y_dividendperstock',
              'y_dividendperstock pct', 'y_roe', 'y_roe pct', 'y_roa', 'y_roa pct', 'y_EBIT', 'y_EBIT pct',
              'y_Net income', 'y_Net income pct']
    y_pred_col = ['y_eps pct']


    from app.c_data_prep.ii_data_prep import data_prep

    data = data_prep(dataset=df_to_use, y_cols=y_cols, iter_cols=time_cols, comp_col=comp_col, keep_raw_cols=[],
                     drop_cols=[])

    qrt_multiplier = 1 if yearly_data else 4

    data.window(input_width=4 * qrt_multiplier, pred_width=qrt_multiplier, shift=1)

    # data.block_static_split(val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=5 * qrt_multiplier * 2,
                             val_time_steps=qrt_multiplier,
                             test_time_steps=qrt_multiplier, shuffle=True)
    # data.single_time_rolling(val_time_steps=1, test_time_steps=1, shuffle=True)

    #data.normalize(method='no')
    data.normalize(method='block')
    #data.normalize(method='time')
    # data.normalize(method='set')

    data.compute()
    print(data)

    data.filter_features(just_include=['3_cap exp', '5_gross margin', '11_FIFO dummy', '8_inventory turnover', '10_inventory to assets pct', '13_depreciation', '14_div per share', '17_ROE', '18_ROE pct chg', '20_CAPEX To Assets last year', '22_debt to equity pct chg', '30_sales to assets', '31_ROA', '54_CF to debt', '57_OpIncome to assets', '2_ROA', '10_Quick Ratio', 'y_eps', 'y_eps pct', 'y_dividendyield', 'y_dividendyield pct', 'y_dividendperstock pct', 'y_EBIT pct', 'y_Net income', '1_inventory_sft_4', '3_cap exp_sft_3', '4_RnD_sft_1', '4_RnD_sft_3', '5_gross margin_sft_1', '5_gross margin_sft_3', '5_gross margin_sft_4', '6_sales admin exp_sft_1', '6_sales admin exp_sft_4', '9_order backlog_sft_1', '2_current ratio_sft_2', '2_current ratio_sft_3', '2_current ratio_sft_4', '8_inventory turnover_sft_1', '10_inventory to assets pct_sft_2', '10_inventory to assets pct_sft_4', '11_inventory_sft_1', '11_inventory_sft_2', '13_depreciation_sft_1', '13_depreciation_sft_2', '14_div per share_sft_1', '14_div per share_sft_2', '14_div per share_sft_3', '17_ROE_sft_2', '18_ROE pct chg_sft_1', '18_ROE pct chg_sft_3', '19_CAPEX To Assets_sft_2', '19_CAPEX To Assets_sft_3', '20_CAPEX To Assets last year_sft_2', '20_CAPEX To Assets last year_sft_4', '21_debt to equity_sft_3', '21_debt to equity_sft_4', '22_debt to equity pct chg_sft_4', '38_pretax income to sales_sft_1', '41_sales to total cash_sft_3', '41_sales to total cash_sft_4', '53_total assets_sft_4', '54_CF to debt_sft_2', '61_Repayment of LT debt _sft_1', '66_Cash div to cash flows_sft_1', '66_Cash div to cash flows_sft_3', '66_Cash div to cash flows_sft_4', '2_ROA_sft_1', '2_ROA_sft_3', '7_Inventory Turnover_sft_2', '7_Inventory Turnover_sft_3', '8_Asset Turnover_sft_1', '8_Asset Turnover_sft_3', '9_Current Ratio_sft_1', '9_Current Ratio_sft_2', '10_Quick Ratio_sft_2', '11_Working Capital_sft_2', '11_Working Capital_sft_3', 'y_eps_sft_1', 'y_eps_sft_2', 'y_eps_sft_3', 'y_eps_sft_4', 'y_eps pct_sft_2', 'y_eps pct_sft_4', 'y_dividendyield_sft_1', 'y_dividendyield pct_sft_1', 'y_dividendyield pct_sft_2', 'y_dividendyield pct_sft_4', 'y_dividendperstock pct_sft_2', 'y_dividendperstock pct_sft_3', 'y_EBIT pct_sft_1', 'y_EBIT pct_sft_2', 'y_Net income_sft_1', 'y_Net income_sft_2'])
    data.filter_y(just_include=y_pred_col)

    out = data['200400_201900']
    data_props = data.get_data_props()

    train_ds, val_ds, test_ds = data.tsds_dataset(out='all', out_dict=None)
    train_np, val_np, test_np = data.np_dataset(out='all', out_dict=None)
    y_dataset, y_params = data.y_dataset(out='all', out_dict=None)
    y_params['y_col'] = y_pred_col
    examples = data.get_examples(example_len=5, example_list=[], y_col=y_pred_col[0])
    examples['pred'] = {}

    val_performance = {}
    test_performance = {}
    examples_predictions = main_run_linear_models(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, val_performance_dict=val_performance, test_performance_dict=test_performance, examples=examples, data_props=data_props)
    #examples['pred'].update(examples_predictions)

    #plot(examples_dict=examples, normalization=False)

    #print("\n\nEvaluate on validation data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str([round(i,2) for i in value]) for key, value in val_performance.items()]), sep='')
    #print("\nEvaluate on test data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str([round(i,2) for i in value]) for key, value in test_performance.items()]), sep='')


