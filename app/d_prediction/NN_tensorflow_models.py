import tensorflow as tf
import z_helpers as my_helpers
import mlflow.keras
import datetime
import pandas as pd

def compile_and_fit(model, train, val, model_name='UNKNOWN', patience=25, MAX_EPOCHS=50):
    tracking_address = my_helpers.get_project_directories(key='tensorboard_logs')
    TBLOGDIR = tracking_address + "/" + model_name

    # Log to MLflow
    mlflow.keras.autolog()  # This is all you need!
    MLFLOW_RUN_NAME = f'{model_name} - {datetime.datetime.now().strftime("%y%m%d_%H%M%S")}'


    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TBLOGDIR, histogram_freq=1)
    model.compile(loss=tf.losses.MeanAbsoluteError(),
                  optimizer=tf.optimizers.Adam(clipnorm=0.001),
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
        summary_table = summary_table.append({"Layer (Type)": layer.name + '(' + layer.__class__.__name__ + ')', "Input Shape": layer.input_shape, "Output Shape": layer.output_shape, "Param #": layer.count_params(), "Dropout": layer.dropout if hasattr(layer, 'dropout') else 'nan', "Bias initializer": layer.bias_initializer._tf_api_names if hasattr(layer, 'bias_initializer') and hasattr(layer.bias_initializer, '_tf_api_names') else 'nan', "Bias regularizer": layer.bias_regularizer if hasattr(layer, 'bias_regularizer') else 'nan'}, ignore_index=True)

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








def main_run_LSTM_models(train_ds, val_ds, test_ds, val_performance_dict, test_performance_dict, data_props, examples=None):

    input_layer_shape = train_ds.element_spec[0].shape[-1]
    target_size = train_ds.element_spec[1].shape[-1]
    timesteps = train_ds.element_spec[0].shape[1]

    linear = tf.keras.Sequential()
    linear.add(tf.keras.layers.BatchNormalization())
    linear.add(tf.keras.layers.LSTM(input_layer_shape, return_sequences=True, input_shape=(timesteps, input_layer_shape)))


    linear.add(tf.keras.layers.BatchNormalization())
    linear.add(tf.keras.layers.LSTM(32, return_sequences=True))
    #linear.add(tf.keras.layers.BatchNormalization())
    #linear.add(tf.keras.layers.LSTM(32))

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




