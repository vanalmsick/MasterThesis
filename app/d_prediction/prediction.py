import os, sys, warnings
import pandas as pd
import mlflow.keras
import mlflow
import shutil, time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')
from app.z_helpers import helpers as my_helpers
#from app.d_prediction.baseline_models import main_run_baseline_models as baseline_models
from app.d_prediction.statistical_models import main_run_linear_models, main_run_statistical_models
from app.d_prediction.NN_tensorflow_models import main_run_LSTM_models
#from app.d_prediction.ML_xxx_models import main as ML_xxx_models





def error_overview(y_pred, y_true):
    from scipy import stats
    data = np.array(y_pred).squeeze() - np.array(y_true).squeeze()
    out = [[('-inf', -2), stats.percentileofscore(data, -2)],
           [(-2,-1.5), stats.percentileofscore(data, -1.5) - stats.percentileofscore(data, -2)],
           [(-1.5,-1), stats.percentileofscore(data, -1) - stats.percentileofscore(data, -1.5)],
           [(-1,-0.5), stats.percentileofscore(data, -0.5) - stats.percentileofscore(data, -1)],
           [(-0.5,-0.25), stats.percentileofscore(data, -0.25) - stats.percentileofscore(data, -0.5)],
           [(-0.25,-0.15), stats.percentileofscore(data, -0.15) - stats.percentileofscore(data, -0.25)],
           [(-0.15,-0.10), stats.percentileofscore(data, -0.10) - stats.percentileofscore(data, -0.15)],
           [(-0.10,-0.05), stats.percentileofscore(data, -0.05) - stats.percentileofscore(data, -0.10)],
           [(-0.05,-0.025), stats.percentileofscore(data, -0.025) - stats.percentileofscore(data, -0.05)],
           [(-0.025,-0.01), stats.percentileofscore(data, -0.01) - stats.percentileofscore(data, -0.025)],
           [(-0.01,0), stats.percentileofscore(data, 0) - stats.percentileofscore(data, -0.01)],
           [(0,0.01), stats.percentileofscore(data, 0.01) - stats.percentileofscore(data, 0)],
           [(0.01,0.025), stats.percentileofscore(data, 0.025) - stats.percentileofscore(data, 0.01)],
           [(0.025,0.05), stats.percentileofscore(data, 0.05) - stats.percentileofscore(data, 0.025)],
           [(0.05,0.1), stats.percentileofscore(data, 0.1) - stats.percentileofscore(data, 0.05)],
           [(0.10,0.15), stats.percentileofscore(data, 0.15) - stats.percentileofscore(data, 0.10)],
           [(0.15,0.25), stats.percentileofscore(data, 0.25) - stats.percentileofscore(data, 0.15)],
           [(0.25,0.5), stats.percentileofscore(data, 0.5) - stats.percentileofscore(data, 0.25)],
           [(0.5,1), stats.percentileofscore(data, 1) - stats.percentileofscore(data, 0.5)],
           [(1,1.5), stats.percentileofscore(data, 1.5) - stats.percentileofscore(data, 1)],
           [(1.5,2), stats.percentileofscore(data, 2) - stats.percentileofscore(data, 1.5)],
           [(2, 'inf'), 1 - stats.percentileofscore(data, 2)]]
    out = np.array(out).T






def plot(examples_dict, normalization=True, y_pct=True):

    color_codes = ['#ff7f0e', '#58D68D', '#A569BD', '#40E0D0', '#922B21', '#CCCCFF', '#0E6655', '#1A5276']

    examples_len = examples_dict['examples_num']
    examples_comp = examples_dict['company'].tolist()

    y_hist = examples_dict['y_hist'].tolist()
    y_true = examples_dict['y_true'].tolist()

    t_idx = examples_dict['t_idx']
    time_steps = examples_dict['time_step']

    y_pred = []
    for i in range(examples_len):
        tmp_dict = {}
        for key, values in examples_dict['pred'].items():
            if len(examples_dict['pred'][key].shape) == 3:
                tmp_data = examples_dict['pred'][key][i, -1, :].tolist()
            elif len(examples_dict['pred'][key].shape) == 2:
                tmp_data = examples_dict['pred'][key][i, :].tolist()
            else:
                raise Exception(f'Strange number of {len(examples_dict["pred"][key].shape)} dimensions. Please check final model output of {key}!')
            tmp_dict[key] = tmp_data
        y_pred.append(tmp_dict)


    if len(examples_dict['pred']) > len(color_codes):
        raise Exception('Too few color codes defined. Please extend color_codes list!')


    fig = plt.figure(figsize=(12, 3 * examples_len))

    for i, time_step, comp, y_hist, y_true, t_idx, y_pred in zip(range(examples_len), time_steps, examples_comp, y_hist, y_true, t_idx, y_pred):
        x_t = [int(str(i)[:-2]) + (int(str(i)[-2:]) / 4) for i in t_idx]

        plt.subplot(examples_len, 1, i + 1)
        plt.ylabel(f'comp {comp} [{i}]')

        if normalization and 'norm_param' in examples_dict:
            scale = 1000000
            mean = examples_dict['norm_param'][i]['mean'][examples_dict['y_cols'][0]]
            std = examples_dict['norm_param'][i]['std'][examples_dict['y_cols'][0]]
        else:
            scale = 1
            mean = 0
            std = 1


        y_true_real = np.array(y_true) * std + mean
        y_hist_real = np.array(y_hist) * std + mean

        x_true = x_t[-len(y_true_real):]
        x_hist = x_t[:len(y_hist_real)]

        plt.plot(x_hist, (y_hist_real / scale), label='Historical', marker='.', zorder=-10)
        plt.plot([x_hist[-1]] + x_true, (np.array([y_hist_real.tolist()[-1]] + y_true_real.tolist()) / scale), label='True', marker='.', c='#A8A8A8', zorder=-11)

        j = 0
        for model, pred_y in y_pred.items():
            y_pred_real = np.array(pred_y) * std + mean
            plt.scatter(x_true[:len(y_pred_real)], (y_pred_real / scale), marker='X', edgecolors='k', label=f'{model} predictions', c=color_codes[j], s=64)
            j += 1

        if i == 0:
            plt.legend()

        #if i == examples_len - 1:
        #    plt.xticks(x_t, rotation='vertical')

        if y_pct:
            warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator")
            vals = fig.get_axes()[i].get_yticks()
            fig.get_axes()[i].set_yticklabels(['{:,.2%}'.format(x) for x in vals])

        fig.get_axes()[i].xaxis.set_major_locator(MaxNLocator(integer=True))


    plt.show()





def run_all_models(train_ds, val_ds, test_ds, train_np, val_np, test_np, y_dataset, y_params, examples, data_props):
    val_performance = {}
    test_performance = {}

    main_run_linear_models(train_ds, val_ds, test_ds, val_performance_dict=val_performance,
                                                              test_performance_dict=test_performance, examples=examples,
                                                              data_props=data_props)


    """
    tmp_exampeles = baseline_models.main_run_baseline_models(train_ds, val_ds, test_ds, data=data, pred_length=4,
                                                             val_performance_dict=val_performance,
                                                             test_performance_dict=test_performance, examples=examples,
                                                             data_props=data_props)
    examples['pred'].update(tmp_exampeles)
    
    tmp_exampeles = statistical_models.main_run_linear_models(train_ds, val_ds, test_ds,
                                                              val_performance_dict=val_performance,
                                                              test_performance_dict=test_performance, examples=examples,
                                                              data_props=data_props)
    examples['pred'].update(tmp_exampeles)
    

    #tmp_exampeles = main_run_linear_models(train_ds, val_ds, test_ds, val_performance_dict=val_performance, test_performance_dict=test_performance, examples=examples, data_props=data_props)

    main_run_statistical_models(y_dataset, y_params, data_props=data_props, examples=None)
    
    tmp_exampeles = main_run_LSTM_models(train_ds, val_ds, test_ds,
                                                              val_performance_dict=val_performance,
                                                              test_performance_dict=test_performance, examples=examples,
                                                              data_props=data_props)
    examples['pred'].update(tmp_exampeles)
    """
    return val_performance, test_performance







if __name__ == '__main__':
    # Working directory must be the higher .../app folder
    from app.z_helpers import helpers as my
    my.convenience_settings()

    ### run tensorboard
    # MLFLOW_TRACKING_URI = 'file:' + '/Users/vanalmsick/Workspace/MasterThesis/cache/MLflow'  # with local tracking serveer
    #MLFLOW_TRACKING_URI = 'http://0.0.0.0:5000'  # with remote tracking server with registry
    #mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
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
    y_pred_col = ['y_eps']


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
    examples = data.get_examples(example_len=5, example_list=[])
    examples['pred'] = {}

    val_performance, test_performance = run_all_models(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, train_np=train_np, val_np=val_np, test_np=test_np, y_dataset=y_dataset, y_params=y_params, examples=examples, data_props=data_props)

    plot(examples_dict=examples, normalization=False)

    print("\n\nEvaluate on validation data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str([round(i,2) for i in value]) for key, value in val_performance.items()]), sep='')
    print("\nEvaluate on test data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str([round(i,2) for i in value]) for key, value in test_performance.items()]), sep='')


