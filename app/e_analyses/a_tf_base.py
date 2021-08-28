import datetime
import os
import warnings

import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
import json

# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')
from app.z_helpers import helpers as my_helpers

from app.d_prediction.NN_tensorflow_models import TF_ERROR_METRICS
from app.d_prediction.NN_tensorflow_models import compile_and_fit, evaluate_model, CustomMeanDirectionalAccuracy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import math
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss


def just_good_features(train_X, train_y=None):
    # drop columns and levae one with high correlation
    corr_matrix = train_X.corr()
    corr_matrix.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

    columns = np.full((corr_matrix.shape[0],), True, dtype=bool)
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[0]):
            if corr_matrix.iloc[i, j] >= 0.9:
                if columns[j]:
                    columns[j] = False

    columns = corr_matrix.index[columns].to_list()

    return columns






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


def createmodel(n_layers, first_layer_nodes, last_layer_nodes, activation_func, input_size, output_size, layer_type='dense', compile=False, model_name=None, **kwargs):
    model = Sequential()
    n_nodes = FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes)
    for i in range(1, n_layers):
        if i == 1:
            if layer_type == 'dense':
                model.add(Dense(first_layer_nodes, input_dim=input_size, activation=activation_func))
            elif layer_type == 'lstm':
                model.add(tf.keras.layers.LSTM(first_layer_nodes, return_sequences=True , input_shape=input_size, activation=activation_func))
            else:
                raise Exception(f'Unknown layer type {layer_type}')
        else:
            if layer_type == 'dense':
                model.add(Dense(n_nodes[i - 1], activation=activation_func))
            elif layer_type == 'lstm':
                model.add(tf.keras.layers.LSTM(first_layer_nodes, return_sequences=True , activation=activation_func))
            else:
                raise Exception(f'Unknown layer type {layer_type}')

    # Finally, the output layer should have a single node in binary classification
    model.add(Dense(output_size, activation=activation_func))

    if compile:
        model.compile(loss=tf.losses.MeanAbsoluteError(), optimizer=tf.optimizers.Adam(), metrics=TF_ERROR_METRICS)

    return model


def main_run_linear_models(train_ds, val_ds, test_ds, data_props, max_backlooking=None, layer_type='dense', activation_funcs=['sigmoid', 'relu', 'tanh'], max_serach_iterations=200, NN_max_depth=3, MAX_EPOCHS=800, patience=25, model_name='linear', examples=None, return_permutation_importances=True, redo_serach_best_model=False):
    mlflow.set_experiment(model_name)
    experiment_date_time = int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    flatten_input = True if layer_type == 'dense' else False


    def _extract_just_important_data_props(data_props):
        kwargs = {}
        kwargs['dataset_cols_X_just_these'] = data_props['third_filter']['cols_just_these']
        kwargs['dataset_cols_X_exclude'] = data_props['third_filter']['cols_drop']
        kwargs['dataset_cols_y'] = data_props['third_filter']['y_cols_just_these']
        kwargs['dataset_hash_input'] = int(data_props['first_step']['dataset'])
        kwargs['dataset_hash_first'] = data_props['first_step_data_hash']
        kwargs['dataset_hash_second'] = data_props['second_step_data_hash']
        kwargs['dataset_split_method'] = data_props['second_step']['split_method']
        kwargs['dataset_split_steps_train'] = data_props['second_step']['split_props']['train_time_steps']
        kwargs['dataset_split_steps_val'] = data_props['second_step']['split_props']['val_time_steps']
        kwargs['dataset_split_steps_test'] = data_props['second_step']['split_props']['test_time_steps']
        kwargs['dataset_iter_step'] = data_props['iter_step']
        kwargs['dataset_normalization'] = data_props['second_step']['normalize_method']
        kwargs['dataset_window_backlooking'] = data_props['first_step']['window_input_width']
        kwargs['dataset_window_prediction'] = data_props['first_step']['window_pred_width']
        kwargs['dataset_window_shift'] = data_props['first_step']['window_shift']
        return kwargs


    def _get_prep_data(train_ds, val_ds, test_ds, flatten=False, keep_last_n_periods='all'):
        # seperate tfds to np
        train_X, train_y = tf.data.experimental.get_single_element(train_ds)
        val_X, val_y = tf.data.experimental.get_single_element(val_ds)
        test_X, test_y = tf.data.experimental.get_single_element(test_ds)
        n_train = train_X.shape[0]
        n_val = val_X.shape[0]
        n_test = test_X.shape[0]

        if keep_last_n_periods != 'all':  # reduce periods
            train_X = train_X[:, -keep_last_n_periods:, :]
            val_X = val_X[:, -keep_last_n_periods:, :]
            test_X = test_X[:, -keep_last_n_periods:, :]

        if flatten:
            train_y = tf.squeeze(train_y)
            val_y = tf.squeeze(val_y)
            test_y = tf.squeeze(test_y)
            train_X = tf.reshape(train_X, (n_train, -1))
            val_X = tf.reshape(val_X, (n_val, -1))
            test_X = tf.reshape(test_X, (n_test, -1))


        # Output / Return
        out = dict(train_ds=tf.data.Dataset.from_tensors((train_X, train_y)),
                   val_ds=tf.data.Dataset.from_tensors((val_X, val_y)),
                   test_ds=tf.data.Dataset.from_tensors((test_X, test_y)),
                   train_X=train_X, train_y=train_y,
                   val_X=val_X, val_y=val_y,
                   test_X=test_X, test_y=test_y,
                   input_shape=train_X.shape[-1],
                   output_shape=1 if len(set(train_y.shape)) == 1 else train_y.shape[-1])

        return out



    def _hp_tranform_param_dict(param_dict):
        new_param_dict = {}
        for key, value in param_dict.items():
            if type(value) == list:
                new_param_dict[key] = hp.choice(key, value)
            elif type(value) == set:
                new_param_dict[key] = hp.uniform(key, *values)
            else:
                new_param_dict[key] = value
        return new_param_dict

    max_backlooking = data_props['first_step']['window_input_width'] if max_backlooking is None else max_backlooking

    param_grid = dict(n_layers=list(range(1, NN_max_depth + 1)),
                      first_layer_nodes=[0] if NN_max_depth == 1 else [128, 64, 32, 16, 8],
                      last_layer_nodes=[0] if NN_max_depth == 1 else [64, 32, 16, 8, 4],
                      activation_func=activation_funcs,
                      backlooking_window=list(range(1, max_backlooking + 1)))
    hp_param_dict = _hp_tranform_param_dict(param_dict=param_grid)
    hp_param_dict['model_name'] = model_name
    hp_param_dict['data_props'] = data_props
    hp_param_dict['layer_type'] = layer_type


    def _optimize_objective(*args, **kwargs):
        if args != ():
            kwargs = args[0]  # if positional arguments expect first to be dictionary with all kwargs
        if type(kwargs) != dict:
            raise Exception(f'kwargs is not  dict - it is {type(kwargs)} with values: {kwargs}')

        backlooking_window = kwargs.pop('backlooking_window')
        n_layers = kwargs.pop('n_layers')
        first_layer_nodes = kwargs.pop('first_layer_nodes')
        last_layer_nodes = kwargs.pop('last_layer_nodes')
        activation_func = kwargs.pop('activation_func')
        return_everything = kwargs.pop('return_everything', False)
        verbose = kwargs.pop('verbose', 0)
        model_name = kwargs.pop('model_name', 'linear')
        data_props = kwargs.pop('data_props')
        layer_type = kwargs.pop('layer_type', 'dense')


        dataset = _get_prep_data(train_ds, val_ds, test_ds, flatten=flatten_input, keep_last_n_periods=backlooking_window)

        now = datetime.datetime.now()
        date_time = str(now.strftime("%y%m%d%H%M%S"))
        model_name = f"{date_time}_{model_name}_w{backlooking_window}_l{n_layers}_a{activation_func}"


        kwargs = dict(model_name=model_name,
                      n_layers=n_layers,
                      first_layer_nodes=first_layer_nodes,
                      last_layer_nodes=last_layer_nodes,
                      activation_func=activation_func,
                      input_size=dataset['input_shape'] if layer_type == 'dense' else tuple(list(train_ds.element_spec[0].shape)[1:]),
                      output_size=dataset['output_shape'],
                      backlooking_window=backlooking_window,
                      layer_type=layer_type)

        model = createmodel(**kwargs)
        history, mlflow_additional_params = compile_and_fit(model=model, train=dataset['train_ds'], val=dataset['val_ds'],
                                                            MAX_EPOCHS=MAX_EPOCHS, patience=patience, model_name=model_name,
                                                            verbose=verbose)

        # Get all data props for documentation in MLflow
        kwargs.update(_extract_just_important_data_props(data_props))
        kwargs['run'] = experiment_date_time
        mlflow_additional_params['kwargs'] = kwargs

        train_performance = dict(zip(model.metrics_names, evaluate_model(model=model, tf_data=dataset['train_ds'])))
        val_performance = dict(zip(model.metrics_names, evaluate_model(model=model, tf_data=dataset['val_ds'])))
        test_performance = dict(zip(model.metrics_names, evaluate_model(model=model, tf_data=dataset['test_ds'], mlflow_additional_params=mlflow_additional_params)))
        mlflow_additional_params['data_props'] = data_props

        # Only save model if close to 15% best models
        try:
            best_loss = float(trials.best_trial['result']['loss'])
            current_loss = min(history.history['val_loss'])
            if current_loss <= best_loss * (1 + 0.15):
                save_model = True
            else:
                save_model = False
        except:
            save_model = True
        mlflow_saved = my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params, save_model=save_model)

        tf.keras.backend.clear_session()

        return_metrics = dict(loss=val_performance['loss'], all_metrics={'train': train_performance, 'val': val_performance, 'test': test_performance},
                              status=STATUS_OK, mlflow=mlflow_saved, model_name=model_name)

        if return_everything:
            return_metrics['model'] = model
            return_metrics['history'] = history

        return return_metrics


    ###### Get old best model records ######

    storage_file_path = os.path.join(my_helpers.get_project_directories(key='cache_dir'), 'storage_best_model.json')
    if not os.path.exists(storage_file_path):
        best_model_storage = {}
    else:
        with open(storage_file_path) as json_file:
            best_model_storage = json.load(json_file)


    ######## Search for best model ########

    if redo_serach_best_model or model_name not in best_model_storage or data_props['iter_step'] not in best_model_storage[model_name]:
        warnings.filterwarnings('ignore')
        trials = Trials()
        best = fmin(fn=_optimize_objective,
                    space=hp_param_dict,
                    algo=tpe.suggest,
                    max_evals=max_serach_iterations,
                    trials=trials,
                    early_stop_fn=no_progress_loss(iteration_stop_count=int(max_serach_iterations/4), percent_increase=0.025))
        warnings.simplefilter('always')

        # getting all parameters for best model storage
        mlflow_best_model = trials.best_trial['result']['mlflow']
        best_params = {}
        for key, idx in best.items():
            best_params[key] = param_grid[key][idx]

        coef_names_ = list(data_props['look_ups']['out_lookup_col_name']['X'].keys())
        coef_names_ = coef_names_ + [col + f'_sft_{i}' for i in range(1, best_params['backlooking_window']) for col in coef_names_]


        # Saving best model to storage
        if model_name not in best_model_storage:
            best_model_storage[model_name] = {}
        if data_props['iter_step'] not in best_model_storage[model_name]:
            best_model_storage[model_name][data_props['iter_step']] = {'best_model': {'result': {'loss': 10 ** 10}}, 'history': {}}

        best_model_param = dict(result={'loss': trials.best_trial['result']['loss'],
                                        'all_metrics': trials.best_trial['result']['all_metrics']},
                                model_name=trials.best_trial['result']['model_name'],
                                model_id=trials.best_trial['result']['mlflow']['model_id'],
                                run_id=experiment_date_time,
                                input_coefs=coef_names_,
                                path_saved_model=trials.best_trial['result']['mlflow']['saved_model_path'],
                                status=trials.best_trial['result']['status'],
                                params=best_params,
                                data=_extract_just_important_data_props(data_props))

        best_model_storage[model_name][data_props['iter_step']]['history'][experiment_date_time] = best_model_param
        if trials.best_trial['result']['loss'] < best_model_storage[model_name][data_props['iter_step']]['best_model']['result']['loss']:
            best_model_storage[model_name][data_props['iter_step']]['best_model'] = best_model_param

        with open(storage_file_path, 'w') as outfile:
            json.dump(best_model_storage, outfile)

    else:
        # Get best model from storage
        best_model_param = best_model_storage[model_name][data_props['iter_step']]['best_model']




    ######## Get Best model again ########
    best_model = tf.keras.models.load_model(best_model_param['path_saved_model'])
    best_model.compile(loss=tf.losses.MeanAbsoluteError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError(), CustomMeanDirectionalAccuracy(), tf.losses.Huber(), tf.metrics.MeanAbsolutePercentageError(), tf.metrics.MeanSquaredError(), tf.metrics.MeanSquaredLogarithmicError()])
    print('Best model is:', best_model_param)

    out = dict(best_model_param)


    ####### Get examples for plotting #######
    if examples is not None:
        example_X = examples['X']
        periods = best_model_param['params']['backlooking_window']
        if layer_type == 'dense':
            example_X = tf.data.Dataset.from_tensors(np.reshape(example_X[:, -periods:, :], (example_X.shape[0], -1)))
        else:
            example_X = tf.data.Dataset.from_tensors(example_X)
        out['examples_pred_y'] = best_model.predict(example_X)



    ###### For 1 layer dense/linear models get coef & p-values ######
    if NN_max_depth == 1 and isinstance(best_model.layers[0], tf.keras.layers.Dense):
        # Get coefs
        intercept_ = best_model.layers[0].bias.numpy()
        coef_ = best_model.layers[0].weights[0].numpy()
        out['coef_'] = pd.Series(dict(zip(['intercept_'] + best_model_param['input_coefs'], intercept_.tolist() + coef_.squeeze().tolist())))

        dataset = _get_prep_data(train_ds, val_ds, test_ds, flatten=True, keep_last_n_periods=best_model_param['params']['backlooking_window'])

        # get p-values
        import app.e_analyses.my_custom_pvalue_calc as my_p_lib

        out['p_values'] = {}
        for data_set in ['train', 'val', 'test']:
            y_pred = best_model.predict(dataset[f'{data_set}_X'])
            y_pred = np.reshape(y_pred, (-1, 1))
            try:
                p_values = my_p_lib.coef_pval(dataset[f'{data_set}_X'], dataset[f'{data_set}_y'], coef_, intercept_, y_pred)
                p_values = pd.Series(dict(zip(best_model_param['input_coefs'], p_values)))
                out['p_values'][data_set] = p_values
            except:
                warnings.warn("P-Values: ValueError: Input contains infinity or nan.")
                out['p_values'][data_set] = pd.Series(dict(zip(best_model_param['input_coefs'], ['error'] * len(best_model_param['input_coefs']))))
        out['p_values'] = pd.DataFrame(out['p_values'])


    ##### Get Column Feature Importance #####
    if return_permutation_importances:
        if 'feature_importance' in best_model_param:
            out['feature_importance'] = best_model_param['feature_importance']

        else:
            import eli5
            from eli5.sklearn import PermutationImportance

            sklearn_model = KerasRegressor(build_fn=best_model)
            sklearn_model.model = best_model

            dataset = _get_prep_data(train_ds, val_ds, test_ds, flatten=flatten_input, keep_last_n_periods=best_model_param['params']['backlooking_window'])

            out['feature_importance'] = {}
            for data_set in ['train', 'val']:
                # Calculate actual FeatureImporttance
                try:
                    perm = PermutationImportance(sklearn_model, cv='prefit').fit(dataset[f'{data_set}_X'].numpy(), np.reshape(dataset[f'{data_set}_y'].numpy(), (-1, 1)))
                    feature_importances = eli5.format_as_dataframe(eli5.explain_weights(perm, feature_names=best_model_param['input_coefs'], top=10 ** 10))
                    out['feature_importance'][data_set] = feature_importances.set_index('feature').to_dict()
                except:
                    warnings.warn("PermutationImportance: ValueError: Input contains infinity or a value too large for dtype('float16').")

            if out['feature_importance'] != {}:
                best_model_param['feature_importance'] = out['feature_importance']
                best_model_storage[model_name][data_props['iter_step']]['best_model']['feature_importance'] = out[ 'feature_importance']
                best_model_storage[model_name][data_props['iter_step']]['history'][experiment_date_time][ 'feature_importance'] = out['feature_importance']

                with open(storage_file_path, 'w') as outfile:
                    json.dump(best_model_storage, outfile)


    out['status'] = 'ok'
    return out


    # ToDo: Use Paper Factors only
    # ToDo: Add LogisticRegression -> Linear: linear() Logit: sigmod() activation functoion

    # ToDo: Feature selection - feature importance score
    # ToDo: Add normalize GridSearch
    # ToDo: Add other normalization techniques
    # ToDo: Add other parameters for GridSearch
    # ToDo: Differnt Layer architectures


def median_scaling(train_ds, val_ds, test_ds, y_col_idx):
    train_X, train_y = tf.data.experimental.get_single_element(train_ds)
    val_X, val_y = tf.data.experimental.get_single_element(val_ds)
    test_X, test_y = tf.data.experimental.get_single_element(test_ds)
    train_X, train_y = train_X.numpy(), train_y.numpy()
    val_X, val_y = val_X.numpy(), val_y.numpy()
    test_X, test_y = test_X.numpy(), test_y.numpy()

    from sklearn import preprocessing
    scaler_X = preprocessing.RobustScaler(with_centering=False, quantile_range=(0.02, 0.98)).fit(train_X.reshape((-1, train_X.shape[-1])))
    #scaler_y = preprocessing.RobustScaler(with_centering=False, quantile_range=(0.02, 0.98)).fit(train_y[:, -1, :])
    #train_y[:, 0, :] = scaler_y.transform(train_y[:, 0, :])
    #val_y[:, 0, :] = scaler_y.transform(val_y[:, 0, :])
    #test_y[:, 0, :] = scaler_y.transform(test_y[:, 0, :])
    for i in range(train_X.shape[1]):
        train_X[:, i, :] = scaler_X.transform(train_X[:, i, :])
        val_X[:, i, :] = scaler_X.transform(val_X[:, i, :])
        test_X[:, i, :] = scaler_X.transform(test_X[:, i, :])

    train_ds = tf.data.Dataset.from_tensors((train_X, train_y))
    val_ds = tf.data.Dataset.from_tensors((val_X, val_y))
    test_ds = tf.data.Dataset.from_tensors((test_X, test_y))

    return train_ds, val_ds, test_ds


def run_model_acorss_time(data_obj, model_name, activation_funcs, y_col, layer_type='dense', max_serach_iterations=200, redo_serach_best_model=False, max_backlooking=None, NN_max_depth=3, MAX_EPOCHS=800, patience=25, example_len=5, example_list=[], export_results=False):

    results_storage = {}

    def _reformat_DF(df, head):
        df = df.append(pd.Series(df.columns.tolist(), name='columns', index=df.columns.tolist()))
        df = df.append(pd.Series([head] * len(df.columns.tolist()), name='time_step', index=df.columns.tolist()))
        df.columns = [f'{head}_{col}' for col in df.columns.tolist()]
        return df


    for out in [data_obj['200000_201500'], data_obj['200100_201600'], data_obj['200200_201700'], data_obj['200300_201800'], data_obj['200400_201900'], data_obj['200500_202000']]:
        print('Time-step:', out['iter_step'])
        train_ds, val_ds, test_ds = data_obj.tsds_dataset(out='all', out_dict=out)
        train_ds, val_ds, test_ds = median_scaling(train_ds, val_ds, test_ds, y_col_idx=out['columns_lookup']['X'][y_col])
        examples = data_obj.get_examples(example_len=example_len, example_list=example_list, y_col=y_col)
        examples['pred'] = {}
        data_props = data_obj.get_data_props()

        results = main_run_linear_models(train_ds, val_ds, test_ds, data_props, max_backlooking=max_backlooking, layer_type=layer_type, activation_funcs=activation_funcs, max_serach_iterations=max_serach_iterations, NN_max_depth=NN_max_depth, MAX_EPOCHS=MAX_EPOCHS, patience=patience, model_name=model_name, examples=examples, redo_serach_best_model=redo_serach_best_model, return_permutation_importances=True)
        if results['status'] == 'ok':
            examples['pred'][model_name] = results['examples_pred_y']

            final_results = {}

            model_props = {'model_name': results['model_name'],
                           'model_id': results['model_id'],
                           'run_id': results['run_id'],
                           'val_loss': results['result']['loss'],
                           'path_saved_model': results['path_saved_model'],
                           'status': results['status']}
            model_props.update(results['params'])
            model_props.update(results['data'])
            model_props = pd.Series(model_props)
            final_results['model'] = model_props


            if 'p_values' in results:
                final_results['pvalues'] = results['p_values']

            if 'coef_' in results:
                final_results['coef'] = results['coef_']


            if 'feature_importance' in results:
                model_fi = pd.DataFrame()
                for data_set, value in results['feature_importance'].items():
                    tmp = pd.DataFrame(value)
                    tmp.columns = [f'{data_set}_{col}' for col in tmp.columns.tolist()]
                    tmp.sort_index(inplace=True)
                    model_fi = pd.concat([model_fi.sort_index(), tmp], axis=1)
                final_results['feature_importance'] = model_fi


            if 'result' in results:
                metrics = pd.DataFrame(results['result']['all_metrics'])
                final_results['error'] = metrics



            # Actual saving as csv
            for key, value in final_results.items():
                if isinstance(value, pd.DataFrame):
                    value = _reformat_DF(value, head=out['iter_step'])
                elif isinstance(value, pd.Series):
                    value = value.rename(out['iter_step'])
                if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                    if key not in results_storage:
                        results_storage[key] = pd.DataFrame()
                    results_storage[key] = pd.concat([results_storage[key].sort_index(), value.sort_index()], axis=1)

                    if export_results != False:
                        results_storage[key].to_csv(export_results + f'results_{model_name}_{key}.csv')
        else:
            warnings.warn(f"Model for {out['iter_step']} did't want to work.")

    return results_storage
