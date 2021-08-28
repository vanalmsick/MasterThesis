import pandas as pd
import tensorflow as tf
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from app.d_prediction.NN_tensorflow_models import compile_and_fit, evaluate_model


# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')
from app.z_helpers import helpers as my_helpers





def main_run_linear_models(train_ds, val_ds, test_ds, val_performance_dict, test_performance_dict, data_props, examples=None):

    linear = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])
    model_name_lin = 'linear'
    history, mlflow_additional_params = compile_and_fit(model=linear, train=train_ds, val=val_ds, MAX_EPOCHS=1000, model_name=model_name_lin)
    mlflow_additional_params['data_props'] = data_props
    val_performance_dict[model_name_lin] = evaluate_model(model=linear, tf_data=val_ds)
    test_performance_dict[model_name_lin] = evaluate_model(model=linear, tf_data=test_ds, mlflow_additional_params=mlflow_additional_params)
    #my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params)


    dense = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ])
    model_name_den = 'dense'
    history, mlflow_additional_params = compile_and_fit(model=dense, train=train_ds, val=val_ds, MAX_EPOCHS=1000, model_name=model_name_den)
    mlflow_additional_params['data_props'] = data_props
    val_performance_dict[model_name_den] = evaluate_model(model=dense, tf_data=val_ds)
    test_performance_dict[model_name_den] = evaluate_model(model=dense, tf_data=test_ds, mlflow_additional_params=mlflow_additional_params)
    #my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params)


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



def logistic_regression_model(train_np, val_np, test_np, val_performance_dict, test_performance_dict, data_props, examples=None):
    def get_R_style_formula(y, x):
        if type(y) == list:
            my_y = y[0]
        else:
            my_y = y
        my_x = ' + '.join(x)
        R_style_formula = my_y + ' ~ ' + my_x + ''
        return R_style_formula

    train_X, train_y = train_np
    test_X, test_y = test_np
    train_X, train_y = train_X[:, -1, :], train_y[:, -1, -1]
    test_X, test_y = test_X[:, -1, :], test_y[:, -1, -1]
    train_y = np.array(train_y >= 0).astype(int)
    test_y = np.array(test_y >= 0).astype(int)


    clf = LogisticRegression(random_state=0).fit(train_X, train_y)
    print('adf')






def main_run_statistical_models(y_dataset, y_params, data_props, examples=None):
    from statsmodels.tsa.arima.model import ARIMA

    y_col = y_params['y_col']

    import pmdarima as pm
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_error

    models_to_try = [(1, 0, 1), (0, 0, 0), (1, 0, 0), (0, 0, 1)]
    results = {}

    for comp in y_dataset:
        for date_key, data in y_dataset[comp].items():
            train_time = date_key[1]
            val_time = date_key[2]
            train_data = data.loc[:train_time][y_col].values
            val_data = data.loc[:val_time][y_col].values
            test_data = data[y_col].values
            val_X = train_y = train_data[:]
            val_y = val_data[-1]
            test_X, test_y = test_data[:-1], test_data[-1]

            bounds = {'start_p': 1, 'start_q': 1,
                      'max_p': 3, 'max_q': 3, 'm': 4,
                      'start_P': 0, 'seasonal': False,
                      'd': 1, 'D': 1, 'trace': True,
                      'error_action': 'ignore',  # don't want to know if an order does not work
                      'suppress_warnings': True,  # don't want convergence warnings
                      'stepwise': True}

            for model in models_to_try:
                model2 = ARIMA(train_y.flatten(), order=(1, 0, 1)).fit()
                forecasts2 = model2.forecast(1)
                error2 = mean_absolute_error(forecasts2, val_y)

                if model not in results:
                    results[model] = []
                results[model].append(error2)

            model = pm.auto_arima(train_y.flatten(), seasonal=False)
            forecasts = model.predict(1)
            error = mean_absolute_error(forecasts, val_y)

    model = pm.auto_arima(train_y.flatten(), **bounds)
    forecasts = model.predict(1)
    print(data)


    import pmdarima as pm
    #for comp_data in train:
    #    model = pm.auto_arima(comp_data, seasonal=False)
    #    forecasts = model.predict(val_y[0].flatten())
    #model = ARIMA(differenced, order=(7, 0, 1))
    #model_fit = model.fit()
    #forecast = model_fit.forecast(steps=7)



