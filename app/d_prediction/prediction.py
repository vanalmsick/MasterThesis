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
#from app.d_prediction.statistical_models import main_run_statistical_models as statistical_models
#from app.d_prediction.NN_tensorflow_models import main_run_LSTM_models as NN_tensorflow_models
#from app.d_prediction.ML_xxx_models import main as ML_xxx_models





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





def run_all_models(train_ds, val_ds, test_ds, train_np, val_np, test_np, examples, data_props):
    val_performance = {}
    test_performance = {}

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
    """

    tmp_exampeles = statistical_models.logistic_regression_model(train_np, val_np, test_np, val_performance_dict=val_performance, test_performance_dict=test_performance, examples=examples, data_props=data_props)

    """
    tmp_exampeles = NN_tensorflow_models.main_run_LSTM_models(train_ds, val_ds, test_ds,
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
    MLFLOW_TRACKING_URI = 'http://0.0.0.0:5000'  # with remote tracking server with registry
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
    yearly_data = False

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

    df_to_use = feature_engerneeing(dataset=df_cleaned, comp_col=comp_col, time_cols=time_cols,
                                    industry_col=industry_col, yearly_data=yearly_data)

    y_cols = ['y_Net income pct']

    from app.c_data_prep.ii_data_prep import data_prep
    data = data_prep(dataset=df_to_use, y_cols=y_cols, iter_cols=time_cols, comp_col=comp_col, keep_raw_cols=[],
                     drop_cols=[])

    data.window(input_width=5 * 4, pred_width=4, shift=1)

    # data.block_static_split(val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=5 * 4 * 2, val_time_steps=4,
                             test_time_steps=4, shuffle=True)
    # data.single_time_rolling(val_time_steps=1, test_time_steps=1, shuffle=True)

    # data.normalize(method='block')
    data.normalize(method='time')
    # data.normalize(method='set')

    data.compute()

    out = data['199904_201804']
    data_props = data.get_data_props()

    train_ds, val_ds, test_ds = data.tsds_dataset(out='all', out_dict=None)
    train_np, val_np, test_np = data.np_dataset(out='all', out_dict=None)
    examples = data.get_examples(example_len=5, example_list=[])
    examples['pred'] = {}

    val_performance, test_performance = run_all_models(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, train_np=train_np, val_np=val_np, test_np=test_np, examples=examples, data_props=data_props)

    plot(examples_dict=examples, normalization=False)

    print("\n\nEvaluate on validation data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str([round(i,2) for i in value]) for key, value in val_performance.items()]), sep='')
    print("\nEvaluate on test data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str([round(i,2) for i in value]) for key, value in test_performance.items()]), sep='')


