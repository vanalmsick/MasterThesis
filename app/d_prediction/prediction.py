import os, sys, math
import pandas as pd
import tensorflow as tf
import shutil, time
import sklearn.decomposition
import matplotlib.pyplot as plt
import numpy as np

import prediction as my_pred

### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import z_helpers as my_helpers
import c_data_prep as my_prep
##################################



def compile_and_fit(model, train, val, model_name='UNKNOWN', patience=50, MAX_EPOCHS=50):
    tracking_address = my_helpers.get_project_directories(key='tensorboard_logs')
    TBLOGDIR = tracking_address + "/" + model_name

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

    return training_history



def plot(examples_dict, normalization=True):

    color_codes = ['#ff7f0e', '#58D68D', '#F5B041', '#A569BD', '#40E0D0']

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
            tmp_data = examples_dict['pred'][key][i, -1, :].tolist()
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






def main_run_baseline_models(train_ds, val_ds, test_ds, val_performance_dict, test_performance_dict, examples=None):

    class Baseline_last_value(tf.keras.Model):
        def __init__(self, label_index=None):
            super().__init__()
            self.label_index = label_index

        def call(self, inputs):
            if self.label_index is None:
                return inputs
            else:
                result = inputs[:, :, self.label_index]
                return result[:, :, tf.newaxis]

        @property
        def name(self):
            return 'baseline_last'


    baseline = Baseline_last_value(label_index=data.latest_out['columns_lookup']['X'][data.dataset_y_col[0]])
    model_name = baseline.name
    history = compile_and_fit(train=train_ds, val=val_ds, model=baseline, MAX_EPOCHS=1, model_name=model_name)
    val_performance_dict[model_name] = baseline.evaluate(val_ds)
    test_performance_dict[model_name] = baseline.evaluate(test_ds)

    if examples is not None:
        example_pred = {}
        example_pred[model_name] = baseline.predict(examples['X_ds'])

        return example_pred



def main_run_LSTM_models(train_ds, val_ds, test_ds, val_performance_dict, test_performance_dict, examples=None):

    input_layer_shape = 267
    target_size = 4
    timesteps = 20

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

    model_name = 'LSTM_'
    history = compile_and_fit(train=train_ds, val=val_ds, model=linear, patience=200, MAX_EPOCHS=1000, model_name=model_name)
    val_performance_dict[model_name] = linear.evaluate(val_ds)
    test_performance_dict[model_name] = linear.evaluate(test_ds)

    if examples is not None:
        example_pred = {}
        example_pred[model_name] = linear.predict(examples['X_ds'])

        return example_pred





if __name__ == '__main__':
    my_helpers.convenience_settings()

    ### run tensorboard
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
    data.normalize(method='time')
    data.compute()


    out = data['200201_201903']
    train_ds, val_ds, test_ds = data.tsds_dataset(out='all', out_dict=None)
    examples = data.get_examples(example_len=5, example_list=[])
    examples['pred'] = {}


    val_performance = {}
    test_performance = {}


    tmp_exampeles = main_run_baseline_models(train_ds, val_ds, test_ds, val_performance_dict=val_performance, test_performance_dict=test_performance, examples=examples)
    examples['pred'].update(tmp_exampeles)

    tmp_exampeles = main_run_LSTM_models(train_ds, val_ds, test_ds, val_performance_dict=val_performance, test_performance_dict=test_performance, examples=examples)
    examples['pred'].update(tmp_exampeles)


    plot(examples_dict=examples, normalization=True)

    print("\n\nEvaluate on validation data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str(value) for key, value in val_performance.items()]), sep='')
    print("\nEvaluate on test data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str(value) for key, value in test_performance.items()]), sep='')


