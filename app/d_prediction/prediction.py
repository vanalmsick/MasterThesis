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



def compile_and_fit(model, train, val, X_train=None, y_train=None, X_val=None, y_val=None, patience=50, model_name='UNKNOWN', MAX_EPOCHS=50):
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
        #training_history = model.fit(X_train, y_train, epochs=MAX_EPOCHS, validation_data=(X_val, y_val), callbacks=[early_stopping, tensorboard_callback])
        training_history = model.fit_generator(train, epochs=MAX_EPOCHS, validation_data=val, callbacks=[early_stopping, tensorboard_callback])
    else:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          patience=patience,
                                                          mode='min',
                                                          restore_best_weights=True)
        #training_history = model.fit(X_train, y_train, epochs=MAX_EPOCHS, callbacks=[early_stopping, tensorboard_callback])
        training_history = model.fit(train, epochs=MAX_EPOCHS, callbacks=[early_stopping, tensorboard_callback])

    return training_history


def plot(y_hist, y_pred, y_true, normalization_param=None, y_name='tr_f_ebit', model_name='Model', examples=0):
    scale = 1000000

    if type(examples) != list:
        examples = [examples]

    plt.figure(figsize=(12, 3 * len(examples)))

    for i in range(len(examples)):
        ex = examples[i]

        plt.subplot(len(examples), 1, i + 1)
        plt.ylabel(f'{y_name} [example {ex}]')

        if normalization_param is None:
            mean = 0
            str = 1
        else:
            mean = normalization_param['mean'][y_name]
            std = normalization_param['std'][y_name]
        y_pred_real = np.array(y_pred[i, -1, :]) * std + mean
        y_true_real = np.array(y_true[i, -1, :]) * std + mean
        y_hist_real = np.array(y_hist[i, -1, :]) * std + mean
        x_hist = [int(i) for i in list(range(0, len(y_hist_real)))]
        x_pred = [int(i) for i in list(range(len(y_hist_real), len(y_hist_real) + len(y_true_real)))]
        plt.plot(x_hist, (y_hist_real / scale), label='Historical', marker='.', zorder=-10)
        plt.scatter(x_pred, (y_true_real / scale), edgecolors='k', label='True', c='#2ca02c', s=64)
        plt.scatter(x_pred, (y_pred_real / scale), marker='X', edgecolors='k', label='Predictions (in-sample/test)', c='#ff7f0e', s=64)

        if i == 0:
            plt.legend()

        xint = range(min(x_hist), math.ceil(max(x_pred)) + 1)
        plt.xticks(xint)

    plt.show()





def plot2(y_hist, y_pred, y_true, normalization_param=None, y_name='tr_f_ebit', model_name='Model', examples=0):
    scale = 1000000
    scale = 1

    if type(examples) != list:
        examples = [examples]

    plt.figure(figsize=(12, 3 * len(examples)))

    for i in range(len(examples)):
        ex = examples[i]

        plt.subplot(len(examples), 1, i + 1)
        plt.ylabel(f'{y_name} [example {ex}]')

        if normalization_param is None:
            mean = 0
            std = 1
        else:
            mean = normalization_param['mean'][y_name]
            std = normalization_param['std'][y_name]
        y_pred_real = np.array(y_pred[i, -1, :]) * std + mean
        y_true_real = np.array(y_true[i, :]) * std + mean
        y_hist_real = np.array(y_hist[i, :]) * std + mean
        x_hist = [int(i) for i in list(range(0, len(y_hist_real)))]
        x_pred = [int(i) for i in list(range(len(y_hist_real), len(y_hist_real) + len(y_true_real)))]
        plt.plot(x_hist, (y_hist_real / scale), label='Historical', marker='.', zorder=-10)
        plt.scatter(x_pred, (y_true_real / scale), edgecolors='k', label='True', c='#2ca02c', s=64)
        plt.scatter(x_pred, (y_pred_real / scale), marker='X', edgecolors='k', label='Predictions (in-sample/test)', c='#ff7f0e', s=64)

        if i == 0:
            plt.legend()

        xint = range(min(x_hist), math.ceil(max(x_pred)) + 1)
        plt.xticks(xint)

    plt.show()







def main_run_baseline_models(train_ds, val_ds, test_ds, val_performance_dict, test_performance_dict):

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



def main_run_LSTM_models(train_ds, val_ds, test_ds, val_performance_dict, test_performance_dict):

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
    history = compile_and_fit(train=train_ds, val=val_ds, model=linear, MAX_EPOCHS=1000, model_name=model_name)
    val_performance_dict[model_name] = linear.evaluate(val_ds)
    test_performance_dict[model_name] = linear.evaluate(test_ds)





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


    val_performance = {}
    test_performance = {}


    main_run_baseline_models(train_ds, val_ds, test_ds, val_performance_dict=val_performance, test_performance_dict=test_performance)

    main_run_LSTM_models(train_ds, val_ds, test_ds, val_performance_dict=val_performance, test_performance_dict=test_performance)


    print("\n\nEvaluate on validation data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str(value) for key, value in val_performance.items()]), sep='')
    print("\nEvaluate on test data:\n", '\n'.join([str(key) + ': ' + (' ' * max(0, 15-len(str(key)))) + str(value) for key, value in test_performance.items()]), sep='')


