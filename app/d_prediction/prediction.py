import os, sys
import pandas as pd
import mlflow.keras
import mlflow
import shutil, time
import matplotlib.pyplot as plt
import numpy as np

### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import z_helpers as my_helpers
import c_data_prep as my_prep
import baseline_models, statistical_models, NN_tensorflow_models, ML_xxx_models
##################################




def plot(examples_dict, normalization=True):

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


    plt.figure(figsize=(12, 3 * examples_len))

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
        plt.scatter(x_true, (y_true_real / scale), edgecolors='k', label='True', c='#2ca02c', s=64)

        j = 0
        for model, pred_y in y_pred.items():
            y_pred_real = np.array(pred_y) * std + mean
            plt.scatter(x_true[:len(y_pred_real)], (y_pred_real / scale), marker='X', edgecolors='k', label=f'{model} predictions', c=color_codes[j], s=64)
            j += 1

        if i == 0:
            plt.legend()

        #if i == examples_len - 1:
        #    plt.xticks(x_t, rotation='vertical')

    plt.show()





def run_all_models(train_ds, val_ds, test_ds, examples, data_props):
    val_performance = {}
    test_performance = {}

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

    tmp_exampeles = NN_tensorflow_models.main_run_LSTM_models(train_ds, val_ds, test_ds,
                                                              val_performance_dict=val_performance,
                                                              test_performance_dict=test_performance, examples=examples,
                                                              data_props=data_props)
    examples['pred'].update(tmp_exampeles)

    return val_performance, test_performance







if __name__ == '__main__':
    my_helpers.convenience_settings()

    ### run tensorboard
    MLFLOW_TRACKING_URI = 'file:' + '/Users/vanalmsick/Workspace/MasterThesis/cache/MLflow'
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    tracking_address = my_helpers.get_project_directories(key='tensorboard_logs')
    try:
        shutil.rmtree(tracking_address)
        time.sleep(10)
    except:
        pass
    os.mkdir(tracking_address)




    data = my_prep.data_prep(dataset='handpicked_dataset2', recache=False, keep_raw_cols='default', drop_cols='default')
    data.window(input_width=5 * 4, pred_width=4, shift=1)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=5 * 4 * 2, val_time_steps=1, test_time_steps=1, shuffle=True)
    data.normalize(method='time')
    data.compute()

    out = data['200201_201903']
    data_props = data.get_data_props()
    #data.export_to_excel()

    train_ds, val_ds, test_ds = data.tsds_dataset(out='all', out_dict=None)
    examples = data.get_examples(example_len=5, example_list=[])
    examples['pred'] = {}

    val_performance, test_performance = run_all_models(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, examples=examples, data_props=data_props)

    plot(examples_dict=examples, normalization=False)

    print("\n\nEvaluate on validation data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str([round(i,2) for i in value]) for key, value in val_performance.items()]), sep='')
    print("\nEvaluate on test data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str([round(i,2) for i in value]) for key, value in test_performance.items()]), sep='')


