import datetime
import os, sys, math
import pandas as pd
import tensorflow as tf
import shutil, time
import sklearn.decomposition
import matplotlib.pyplot as plt
import numpy as np
import mlflow.keras
import keras
import datetime
#from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME

import prediction as my_pred

### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import z_helpers as my_helpers
import c_data_prep as my_prep
##################################



def compile_and_fit(model, train, val, model_name='UNKNOWN', patience=50, MAX_EPOCHS=50):
    tracking_address = my_helpers.get_project_directories(key='tensorboard_logs')
    TBLOGDIR = tracking_address + "/" + model_name

    # Log to MLflow
    mlflow.keras.autolog()  # This is all you need!
    MLFLOW_RUN_NAME = f'{model_name} - {datetime.datetime.now().strftime("%y%m%d_%H%M%S")}'


    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TBLOGDIR, histogram_freq=1)
    model.compile(loss=tf.losses.MeanAbsolutePercentageError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanAbsolutePercentageError(), tf.metrics.MeanSquaredLogarithmicError(), tf.metrics.MeanSquaredError()])

    if val is not None:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min',
                                                          restore_best_weights=True)
        training_history = model.fit(train, epochs=MAX_EPOCHS, validation_data=val, callbacks=[early_stopping, tensorboard_callback])
    else:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          patience=patience,
                                                          mode='min',
                                                          restore_best_weights=True)
        training_history = model.fit(train, epochs=MAX_EPOCHS, callbacks=[early_stopping, tensorboard_callback])

    summary_table = pd.DataFrame(columns=["Layer (Type)", "Input Shape", "Output Shape", "Param #", "Dropout", "Bias initializer", "Bias regularizer"])
    for layer in model.layers:
        summary_table = summary_table.append({"Layer (Type)": layer.name + '(' + layer.__class__.__name__ + ')', "Input Shape": layer.input_shape, "Output Shape": layer.output_shape, "Param #": layer.count_params(), "Dropout": layer.dropout if hasattr(layer, 'dropout') else 'nan', "Bias initializer": layer.bias_initializer._tf_api_names, "Bias regularizer": layer.bias_regularizer}, ignore_index=True)

    mlflow_additional_params = {'layer_df': summary_table,
                                'model_name': model_name,
                                'max_epochs': early_stopping.params['epochs'],
                                'actual_epochs': early_stopping.stopped_epoch,
                                'early_stopped': model.stop_training,
                                'loss': model.loss.name}

    print(model_name)
    print(model.summary())

    return training_history, mlflow_additional_params



def evaluate_model(model, tf_data, mlflow_additional_params=None):
    performance = model.evaluate(tf_data)
    if mlflow_additional_params is not None:
        mlflow_additional_params['metrics_test'] = dict(zip(model.metrics_names, performance))
    return performance



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






def main_run_baseline_models(train_ds, val_ds, test_ds, val_performance_dict, test_performance_dict, data_props, pred_length=4, examples=None):

    class Baseline_last_value(tf.keras.Model):
        def __init__(self, label_index=None):
            super().__init__()
            self.label_index = label_index

        def call(self, inputs):
            if self.label_index is None:
                return inputs
            else:
                result = inputs[:, :, self.label_index]
                result = result[:, :, tf.newaxis]
                final_result = result
                for _ in range(pred_length - 1):
                    final_result = tf.concat((final_result, result), axis=2)
                return final_result

        @property
        def name(self):
            return 'baseline_last'


    baseline = Baseline_last_value(label_index=data.latest_out['columns_lookup']['X'][data.dataset_y_col[0]])
    model_name = baseline.name
    history, mlflow_additional_params = compile_and_fit(train=train_ds, val=val_ds, model=baseline, MAX_EPOCHS=1, model_name=model_name)
    mlflow_additional_params['data_props'] = data_props
    val_performance_dict[model_name] = evaluate_model(model=baseline, tf_data=val_ds)
    test_performance_dict[model_name] = evaluate_model(model=baseline, tf_data=test_ds, mlflow_additional_params=mlflow_additional_params)
    my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params)



    class Baseline_4last_value(tf.keras.Model):
        def __init__(self, label_index=None):
            super().__init__()
            self.label_index = label_index

        def call(self, inputs):
            if self.label_index is None:
                return inputs
            else:
                result = inputs[:, -pred_length:, self.label_index]
                result = result[:, tf.newaxis, :]
                return result

        @property
        def name(self):
            return 'baseline_4last'


    baseline4 = Baseline_4last_value(label_index=data.latest_out['columns_lookup']['X'][data.dataset_y_col[0]])
    model_name4 = baseline4.name
    history, mlflow_additional_params = compile_and_fit(train=train_ds, val=val_ds, model=baseline4, MAX_EPOCHS=1, model_name=model_name4)
    mlflow_additional_params['data_props'] = data_props
    val_performance_dict[model_name4] = evaluate_model(model=baseline4, tf_data=val_ds)
    test_performance_dict[model_name4] = evaluate_model(model=baseline4, tf_data=test_ds, mlflow_additional_params=mlflow_additional_params)
    my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params)


    if examples is not None:
        example_pred = {}
        example_pred[model_name] = baseline.predict(examples['X_ds'])
        example_pred[model_name4] = baseline4.predict(examples['X_ds'])

        return example_pred




def main_run_linear_models(train_ds, val_ds, test_ds, val_performance_dict, test_performance_dict, data_props, examples=None):

    linear = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])
    model_name_lin = 'linear'
    history, mlflow_additional_params = compile_and_fit(model=linear, train=train_ds, val=val_ds, MAX_EPOCHS=100, model_name=model_name_lin)
    mlflow_additional_params['data_props'] = data_props
    val_performance_dict[model_name_lin] = evaluate_model(model=linear, tf_data=val_ds)
    test_performance_dict[model_name_lin] = evaluate_model(model=linear, tf_data=test_ds, mlflow_additional_params=mlflow_additional_params)
    my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params)


    dense = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ])
    model_name_den = 'dense'
    history, mlflow_additional_params = compile_and_fit(model=dense, train=train_ds, val=val_ds, MAX_EPOCHS=100, model_name=model_name_den)
    mlflow_additional_params['data_props'] = data_props
    val_performance_dict[model_name_den] = evaluate_model(model=dense, tf_data=val_ds)
    test_performance_dict[model_name_den] = evaluate_model(model=dense, tf_data=test_ds, mlflow_additional_params=mlflow_additional_params)
    my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params)


    multi_step_dense = tf.keras.Sequential([
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([1, -1]),
        ])
    model_name_multi_den = 'multi_dense'
    history, mlflow_additional_params = compile_and_fit(model=multi_step_dense, train=train_ds, val=val_ds, MAX_EPOCHS=100, model_name=model_name_multi_den)
    mlflow_additional_params['data_props'] = data_props
    val_performance_dict[model_name_multi_den] = evaluate_model(model=multi_step_dense, tf_data=val_ds)
    test_performance_dict[model_name_multi_den] = evaluate_model(model=multi_step_dense, tf_data=test_ds, mlflow_additional_params=mlflow_additional_params)
    my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params)


    if examples is not None:
        example_pred = {}
        example_pred[model_name_lin] = linear.predict(examples['X_ds'])
        example_pred[model_name_den] = dense.predict(examples['X_ds'])
        example_pred[model_name_multi_den] = multi_step_dense.predict(examples['X_ds'])

        return example_pred







def main_run_statistical_models(train_ds, val_ds, test_ds, val_performance_dict, test_performance_dict, data_props, examples=None):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(differenced, order=(7, 0, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=7)





def main_run_LSTM_models(train_ds, val_ds, test_ds, val_performance_dict, test_performance_dict, data_props, examples=None):

    input_layer_shape = 267
    target_size = 4
    timesteps = 20

    linear = tf.keras.Sequential()
    #linear.add(tf.keras.layers.BatchNormalization())
    linear.add(tf.keras.layers.LSTM(input_layer_shape, return_sequences=True, input_shape=(timesteps, input_layer_shape)))
    #linear.add(tf.keras.layers.BatchNormalization())
    #linear.add(tf.keras.layers.LSTM(32, return_sequences=True))
    #linear.add(tf.keras.layers.BatchNormalization())
    #linear.add(tf.keras.layers.LSTM(32))
    #linear.add(tf.keras.layers.BatchNormalization())
    linear.add(tf.keras.layers.Dense(target_size, activation='softmax'))

    """
    linear = tf.keras.Sequential([
        # tf.keras.layers.LayerNormalization(axis=[1,2]),
        # tf.keras.layers.Dense(int(input_layer_shape*0.75)),
        # tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.LSTM(input_layer_shape, return_sequences=True),
        # tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(timesteps, data_dim)),
        # tf.keras.layers.LSTM(128, return_sequences=True),
        # tf.keras.layers.LSTM(int(input_layer_shape * 0.75), return_sequences=False),
        # tf.keras.layers.Dropout(rate=0.2),
        # tf.keras.layers.LSTM(input_layer_shape, return_sequences=True),
        # tf.keras.layers.Dense(target_size * 1, kernel_initializer=tf.initializers.ones()),
        tf.keras.layers.Dense(target_size * 1, kernel_initializer=tf.initializers.zeros())
    ])
    """

    model_name = 'LSTM'
    history, mlflow_additional_params = compile_and_fit(train=train_ds, val=val_ds, model=linear, patience=200, MAX_EPOCHS=1000, model_name=model_name)
    mlflow_additional_params['data_props'] = data_props
    val_performance_dict[model_name] = evaluate_model(model=linear, tf_data=val_ds)
    test_performance_dict[model_name] = evaluate_model(model=linear, tf_data=test_ds, mlflow_additional_params=mlflow_additional_params)
    my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params)


    if examples is not None:
        example_pred = {}
        example_pred[model_name] = linear.predict(examples['X_ds'])

        return example_pred





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



    data = my_prep.data_prep(dataset='final_data_2', recache=False, keep_raw_cols='default', drop_cols='default')
    data.window(input_width=5 * 4, pred_width=4, shift=1)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=5 * 4 * 2, val_time_steps=1, test_time_steps=1, shuffle=True)
    data.normalize(method='set')
    data.compute()

    out = data['200201_201903']
    data_props = data.get_data_props()

    train_ds, val_ds, test_ds = data.tsds_dataset(out='all', out_dict=None)
    examples = data.get_examples(example_len=5, example_list=[])
    examples['pred'] = {}


    val_performance = {}
    test_performance = {}


    tmp_exampeles = main_run_baseline_models(train_ds, val_ds, test_ds, pred_length=4, val_performance_dict=val_performance, test_performance_dict=test_performance, examples=examples, data_props=data_props)
    examples['pred'].update(tmp_exampeles)

    #tmp_exampeles = main_run_linear_models(train_ds, val_ds, test_ds, val_performance_dict=val_performance, test_performance_dict=test_performance, examples=examples, data_props=data_props)
    #examples['pred'].update(tmp_exampeles)

    tmp_exampeles = main_run_LSTM_models(train_ds, val_ds, test_ds, val_performance_dict=val_performance, test_performance_dict=test_performance, examples=examples, data_props=data_props)
    examples['pred'].update(tmp_exampeles)


    plot(examples_dict=examples, normalization=False)

    print("\n\nEvaluate on validation data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str([round(i,2) for i in value]) for key, value in val_performance.items()]), sep='')
    print("\nEvaluate on test data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str([round(i,2) for i in value]) for key, value in test_performance.items()]), sep='')


