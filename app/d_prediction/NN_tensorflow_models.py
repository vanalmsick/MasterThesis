import tensorflow as tf
import os
import mlflow.keras
import datetime
import pandas as pd
import keras as K

# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')
from app.z_helpers import helpers as my_helpers


import keras


class CustomMeanDirectionalAccuracy(keras.losses.Loss):
    def __init__(self, name="mean_directional_accuracy", rounding_accuracy=10):
        super().__init__(name=name)
        self.rounding = rounding_accuracy

    def call(self, y_true, y_pred):
        base = tf.cast((tf.math.sign(y_true)) == tf.math.sign(y_pred), dtype=tf.float32)

        ## compare rounded and rounded y
        y_pred_rounded = tf.math.round(y_pred * self.rounding) / self.rounding
        y_true_rounded = tf.math.round(y_true * self.rounding) / self.rounding
        add_rounding = tf.cast(y_pred_rounded == y_true_rounded, dtype=tf.float32)


        combined = tf.math.maximum(base, add_rounding)

        return tf.math.reduce_mean(combined)


class CustomPctPosDirection(keras.losses.Loss):
    def __init__(self, name="true_direction", direction='pos', zero_rounding_accuracy=1000000):
        super().__init__(name=f'{name}_{direction}')
        self.direction = direction
        self.rounding = zero_rounding_accuracy

    def call(self, y_true, y_pred=None):
        y_true = tf.math.round(y_true * self.rounding) / self.rounding

        if self.direction == 'pos' or self.direction == 1 or self.direction == 'positive':
            compare_against = tf.ones_like(y_true)
        elif self.direction == 'zero' or self.direction == 0:
            compare_against = tf.zeros_like(y_true)
        elif self.direction == 'neg' or self.direction == -1 or self.direction == 'negative':
            compare_against = tf.ones_like(y_true) * (-1)
        else:
            raise Exception(f'Unknown direction type {self.direction} for CustomPctPosDirection().')

        base = tf.cast(tf.math.sign(y_true) == compare_against, dtype=tf.float32)

        return tf.math.reduce_mean(base)




class CustomElendKerstinError(keras.losses.Loss):
    # JUST APPLICABLE WHEN USING ABSOLUTE DATE NOT PCT CHANGE EARNINGS CHANGES
    def __init__(self, name="elend_kerstin_error", mse_or_mae='mse'):
        super().__init__(name=f'{name}-{mse_or_mae}')
        self.mse_or_mae = mse_or_mae

    def call(self, y_true, y_pred):
        y_0 = tf.zeros_like(y_pred)
        if self.mse_or_mae == 'mse':
            upper = tf.keras.metrics.mean_squared_error(y_true, y_pred)
            lower = tf.keras.metrics.mean_absolute_error(y_true, y_0)
        elif self.mse_or_mae == 'mae':
            upper = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
            lower = tf.keras.metrics.mean_absolute_error(y_true, y_0)
        error = 1 - (upper / lower)
        return error




TF_ERROR_METRICS = [tf.metrics.MeanAbsoluteError(), CustomMeanDirectionalAccuracy(), CustomPctPosDirection(), tf.losses.Huber(), tf.metrics.MeanAbsolutePercentageError(), tf.metrics.MeanSquaredError(), tf.metrics.MeanSquaredLogarithmicError()]

def compile_and_fit(model, train, val, model_name='UNKNOWN', patience=25, MAX_EPOCHS=50, verbose=1):
    tracking_address = my_helpers.get_project_directories(key='tensorboard_logs')
    TBLOGDIR = tracking_address + "/" + model_name

    # Log to MLflow
    mlflow.keras.autolog()  # This is all you need!
    MLFLOW_RUN_NAME = f'{model_name} - {datetime.datetime.now().strftime("%y%m%d_%H%M%S")}'


    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TBLOGDIR, histogram_freq=1)
    model.compile(loss=tf.losses.MeanAbsoluteError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=TF_ERROR_METRICS)

    if val is not None:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min',
                                                          restore_best_weights=True)
        training_history = model.fit(train, epochs=MAX_EPOCHS, validation_data=val, callbacks=[early_stopping, tensorboard_callback], verbose=verbose)
    else:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          patience=patience,
                                                          mode='min',
                                                          restore_best_weights=True)
        training_history = model.fit(train, epochs=MAX_EPOCHS, callbacks=[early_stopping, tensorboard_callback], verbose=verbose)

    summary_table = pd.DataFrame(columns=["Layer (Type)", "Input Shape", "Output Shape", "Param #", "Dropout", "Bias initializer", "Bias regularizer"])
    for layer in model.layers:
        summary_table = summary_table.append({"Layer (Type)": layer.name + '(' + layer.__class__.__name__ + ')', "Input Shape": layer.input_shape, "Output Shape": layer.output_shape, "Param #": layer.count_params(), "Dropout": layer.dropout if hasattr(layer, 'dropout') else 'nan', "Bias initializer": layer.bias_initializer._tf_api_names if hasattr(layer, 'bias_initializer') and hasattr(layer.bias_initializer, '_tf_api_names') else 'nan', "Bias regularizer": layer.bias_regularizer if hasattr(layer, 'bias_regularizer') else 'nan'}, ignore_index=True)

    mlflow_additional_params = {'layer_df': summary_table,
                                'model_type': 'TensorFlow',
                                'history_obj': training_history,
                                'model_name': model_name,
                                'max_epochs': early_stopping.params['epochs'],
                                'actual_epochs': early_stopping.stopped_epoch,
                                'early_stopped': model.stop_training,
                                'loss': model.loss.name}

    if verbose != 0:
        print(model_name)
        print(model.summary())

    return training_history, mlflow_additional_params



def evaluate_model(model, tf_data, mlflow_additional_params=None):
    performance = model.evaluate(tf_data)
    if mlflow_additional_params is not None:
        mlflow_additional_params['metrics_test'] = dict(zip(model.metrics_names, performance))
    return performance








def main_run_LSTM_models(train_ds, val_ds, test_ds, val_performance_dict, test_performance_dict, data_props, examples=None):

    input_layer_shape = train_ds.element_spec[0].shape[-1]
    target_size = train_ds.element_spec[1].shape[-1]
    timesteps = train_ds.element_spec[0].shape[1]

    linear = tf.keras.Sequential()
    #linear.add(tf.keras.layers.BatchNormalization())
    #linear.add(tf.keras.layers.LSTM(input_layer_shape, return_sequences=True, input_shape=(timesteps, input_layer_shape)))


    #linear.add(tf.keras.layers.BatchNormalization())
    #linear.add(tf.keras.layers.LSTM(32, return_sequences=True))


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
    history, mlflow_additional_params = compile_and_fit(train=train_ds, val=val_ds, model=linear, patience=100, MAX_EPOCHS=500, model_name=model_name)
    mlflow_additional_params['data_props'] = data_props
    val_performance_dict[model_name] = evaluate_model(model=linear, tf_data=val_ds)
    test_performance_dict[model_name] = evaluate_model(model=linear, tf_data=test_ds, mlflow_additional_params=mlflow_additional_params)
    my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params)


    if examples is not None:
        example_pred = {}
        example_pred[model_name] = linear.predict(examples['X_ds'])

        return example_pred




