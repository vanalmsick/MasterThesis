import datetime
import os
import warnings

import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf

# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')
from app.z_helpers import helpers as my_helpers

from app.d_prediction.NN_tensorflow_models import compile_and_fit, evaluate_model, MeanDirectionalAccuracy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import math
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss


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


def createmodel(n_layers, first_layer_nodes, last_layer_nodes, activation_func, input_size, output_size, compile=False, **kwargs):
    model = Sequential()
    n_nodes = FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes)
    for i in range(1, n_layers):
        if i == 1:
            model.add(Dense(first_layer_nodes, input_dim=input_size, activation=activation_func))
        else:
            model.add(Dense(n_nodes[i - 1], activation=activation_func))

    # Finally, the output layer should have a single node in binary classification
    model.add(Dense(output_size, activation=activation_func))

    if compile:
        model.compile(loss=tf.losses.MeanAbsoluteError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanAbsolutePercentageError(),
                               tf.metrics.MeanSquaredLogarithmicError(), tf.metrics.MeanSquaredError()])

    return model


def main_run_linear_models(train_ds, val_ds, test_ds, data_props, activation_funcs=['sigmoid', 'relu', 'tanh'], max_serach_iterations=200, NN_max_depth=3, MAX_EPOCHS=800, patience=25, model_name='linear', examples=None, return_permutation_importances=True, redo_serach_best_model=False):

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


    param_grid = dict(n_layers=list(range(1, NN_max_depth + 1)),
                      first_layer_nodes=[0] if NN_max_depth == 1 else [128, 64, 32, 16],
                      last_layer_nodes=[0] if NN_max_depth == 1 else [32, 16, 4],
                      activation_func=activation_funcs,
                      backlooking_window=[1, 2, 3, 4])
    hp_param_dict = _hp_tranform_param_dict(param_dict=param_grid)
    hp_param_dict['model_name'] = model_name


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


        dataset = _get_prep_data(train_ds, val_ds, test_ds, flatten=True, keep_last_n_periods=backlooking_window)

        now = datetime.datetime.now()
        date_time = str(now.strftime("%y%m%d%H%M%S"))
        model_name = f"{date_time}_{model_name}_w{backlooking_window}_l{n_layers}_a{activation_func}"


        kwargs = dict(model_name=model_name,
                      n_layers=n_layers,
                      first_layer_nodes=first_layer_nodes,
                      last_layer_nodes=last_layer_nodes,
                      activation_func=activation_func,
                      input_size=dataset['input_shape'],
                      output_size=dataset['output_shape'],
                      backlooking_window=backlooking_window)

        model = createmodel(**kwargs)
        history, mlflow_additional_params = compile_and_fit(model=model, train=dataset['train_ds'], val=dataset['val_ds'],
                                                            MAX_EPOCHS=MAX_EPOCHS, patience=patience, model_name=model_name,
                                                            verbose=verbose)


        mlflow_additional_params['kwargs'] = kwargs

        train_performance = evaluate_model(model=model, tf_data=dataset['train_ds'])
        val_performance = evaluate_model(model=model, tf_data=dataset['val_ds'])
        test_performance = evaluate_model(model=model, tf_data=dataset['train_ds'], mlflow_additional_params=mlflow_additional_params)

        my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params)

        tf.keras.backend.clear_session()

        return_metrics = dict(loss=val_performance[0],
                              status=STATUS_OK)

        if return_everything:
            return dict(model=model,
                        model_name=model_name,
                        train_history=history,
                        train_performance=train_performance,
                        val_performance=val_performance,
                        test_performance=test_performance)
        else:
            return return_metrics


    ######## Search for best model ########

    if redo_serach_best_model:
        warnings.filterwarnings('ignore')
        trials = Trials()
        best = fmin(fn=_optimize_objective,
                    space=hp_param_dict,
                    algo=tpe.suggest,
                    max_evals=max_serach_iterations,
                    trials=trials,
                    early_stop_fn=no_progress_loss(iteration_stop_count=20, percent_increase=0.05))
        warnings.simplefilter('always')

    else:
        # Hard code best results
        best = {'activation_func': 2, 'backlooking_window': 0, 'first_layer_nodes': 2, 'last_layer_nodes': 1, 'n_layers': 0}

    best_params = {}
    for key, idx in best.items():
        best_params[key] = param_grid[key][idx]
    print('Best Model:', best_params)


    ######## Get Best model again ########

    output = _optimize_objective(**best_params, return_everything=True, verbose=1, model_name=f'best_{model_name}')
    print('Best:', best_params)
    print(output)


    out = dict(model_name=output['model_name'],
               model=output['model'],
               error=pd.DataFrame({'train': output['train_performance'], 'val': output['val_performance'],'test': output['test_performance']}, index=output['model'].metrics_names),
               best_model=pd.Series(best_params))

    if np.isnan(output['train_performance'][0]):
        out['status'] = 'error'
        return out
    else:
        out['status'] = 'ok'


    ####### Example Plotting #######

    if examples is not None:
        example_X = examples['X']
        periods = best_params['backlooking_window']
        example_X = tf.data.Dataset.from_tensors(np.reshape(example_X[:, -periods:, :], (example_X.shape[0], -1)))
        model = output['model']
        out['examples_pred_y'] = model.predict(example_X)



    ###### P-Values if 1 level only ######
    if NN_max_depth == 1:
        intercept_ = output['model'].layers[0].bias.numpy()
        coef_ = output['model'].layers[0].weights[0].numpy()
        coef_names_ = list(data_props['look_ups']['out_lookup_col_name']['X'].keys())
        out['coef_'] = pd.Series(dict(zip(['intercept_'] + coef_names_, intercept_.tolist() + coef_.squeeze().tolist())))

        dataset = _get_prep_data(train_ds, val_ds, test_ds, flatten=True, keep_last_n_periods=best_params['backlooking_window'])


        ###### Get P Values ######
        import app.e_analyses.my_custom_pvalue_calc as my_p_lib

        out['p_values'] = {}
        for data_set in ['train', 'val', 'test']:
            y_pred = output['model'].predict(dataset[f'{data_set}_X'])
            y_pred = np.reshape(y_pred, (-1, 1))
            try:
                p_values = my_p_lib.coef_pval(dataset[f'{data_set}_X'], dataset[f'{data_set}_y'], coef_, intercept_, y_pred)
                p_values = pd.Series(dict(zip(coef_names_, p_values)))
                out['p_values'][data_set] = p_values
            except:
                warnings.warn("P-Values: ValueError: Input contains infinity or nan.")
                out['p_values'][data_set] = pd.Series(dict(zip(coef_names_, ['error'] * len(coef_names_))))
        out['p_values'] = pd.DataFrame(out['p_values'])


        ##### Get Column Feature Importance #####
        if return_permutation_importances:
            import eli5
            from eli5.sklearn import PermutationImportance

            # Reconstruct model as Sklearn KerasRegressor model for Permutation Importance
            best_params['input_size'] = dataset['input_shape']
            best_params['output_size'] = dataset['output_shape']
            best_params['compile'] = True
            my_model = KerasRegressor(build_fn=lambda: createmodel(**best_params))
            my_model.fit(dataset['train_X'].numpy(), np.reshape(dataset['train_y'].numpy(), (-1, 1)), verbose=0, epochs=2)
            # Set/overwrite weights to have exactly the same model as above
            my_model.model.layers[0].set_weights([coef_, intercept_])
            # Validate if TF model same as new
            y_pred = np.reshape(output['model'].predict(dataset['train_X'].numpy()), (-1, 1))
            y_pred_validation = np.reshape(my_model.predict(dataset['train_X'].numpy()), (-1, 1))
            if np.array_equal(y_pred, y_pred_validation) is False:
                raise Exception('Re-fitted sklearn KerasRegressor Model for Permutation Importance is NOT IDENTICAL to TF best fit model!')

            out['feature_importance'] = pd.DataFrame()
            for data_set in ['train', 'val', 'test']:
                # Calculate actual FeatureImporttance
                try:
                    perm = PermutationImportance(my_model, cv='prefit').fit(dataset[f'{data_set}_X'].numpy(), np.reshape(dataset[f'{data_set}_y'].numpy(), (-1, 1)))
                    feature_importances = eli5.format_as_dataframe(eli5.explain_weights(perm, feature_names=coef_names_, top=10 ** 10))
                    out['feature_importance'] = pd.concat([out['feature_importance'].sort_index(), feature_importances.set_index('feature').rename(columns={'weight': f'{data_set}_weight', 'std': f'{data_set}_std'}).sort_index()], axis=1)
                except:
                    warnings.warn("PermutationImportance: ValueError: Input contains infinity or a value too large for dtype('float16').")

    return out


    # ToDo: Use Paper Factors only
    # ToDo: Add LogisticRegression -> Linear: linear() Logit: sigmod() activation functoion

    # ToDo: Feature selection - feature importance score
    # ToDo: Add normalize GridSearch
    # ToDo: Add other normalization techniques
    # ToDo: Add other parameters for GridSearch
    # ToDo: Differnt Layer architectures




def run_model_acorss_time(data_obj, model_name, activation_funcs, y_col, max_serach_iterations=200, redo_serach_best_model=False, NN_max_depth=3, MAX_EPOCHS=800, patience=25, example_len=5, example_list=[], export_results=False):

    results_storage = {}

    def _reformat_DF(df, head):
        df = df.append(pd.Series(df.columns.tolist(), name='columns', index=df.columns.tolist()))
        df = df.append(pd.Series([head] * len(df.columns.tolist()), name='time_step', index=df.columns.tolist()))
        df.columns = [f'{head}_{col}' for col in df.columns.tolist()]
        return df


    for out in [data_obj['200000_201500'], data_obj['200100_201600'], data_obj['200200_201700'], data_obj['200300_201800'], data_obj['200400_201900'], data_obj['200500_202000']]:
        print('Time-step:', out['iter_step'])
        train_ds, val_ds, test_ds = data_obj.tsds_dataset(out='all', out_dict=out)
        examples = data_obj.get_examples(example_len=example_len, example_list=example_list, y_col=y_col)
        examples['pred'] = {}
        data_props = data_obj.get_data_props()

        results = main_run_linear_models(train_ds, val_ds, test_ds, data_props, activation_funcs=activation_funcs, max_serach_iterations=max_serach_iterations, NN_max_depth=NN_max_depth, MAX_EPOCHS=MAX_EPOCHS, patience=patience, model_name=model_name, examples=examples, redo_serach_best_model=redo_serach_best_model, return_permutation_importances=True)
        if results['status'] == 'ok':
            examples['pred'][model_name] = results['examples_pred_y']


            # Get all data props for documentation
            result_data_props = dict(**data_props['first_step'], first_step_data_hash=data_props['first_step_data_hash'])
            result_data_props.update(**data_props['second_step'], second_step_data_hash=data_props['second_step_data_hash'])
            result_data_props.update(**data_props['third_filter'], iter_step=data_props['iter_step'])
            result_data_props.pop('data_hash')
            result_data_props = pd.Series(result_data_props, name=out['iter_step'])
            results['data'] = result_data_props
            results['best_model'] = results['best_model'].append(pd.Series([results['model_name']], index=['model_id']))
            results['best_model'] = results['best_model'].append(pd.Series([model_name], index=['model_name']))

            for key, value in results.items():
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






if __name__ == '__main__':
    import shutil, time

    # Working directory must be the higher .../app folder
    from app.z_helpers import helpers as my
    my.convenience_settings()

    ############################## USER SETTINGS ##############################

    # MLflow model registry location
    MLFLOW_TRACKING_URI = 'file:' + '/Users/vanalmsick/Workspace/MasterThesis/cache/MLflow'  # with local tracking serveer
    #MLFLOW_TRACKING_URI = 'http://0.0.0.0:5000'  # with remote tracking server with registry

    # dataset selection
    dataset_name = 'handpicked_dataset'
    yearly_data = True  # False = quarterly data

    # re-caching
    recache_raw_data = False
    redo_data_cleaning = False

    # data cleaning
    required_filled_cols_before_filling = ['sales', 'roe', 'ebit']
    required_filled_cols_after_filling = []
    drop_threshold_row_pct = 0.4
    drop_threshold_row_quantile = 0.15
    drop_threshold_col_pct = 0
    append_data_quality_col = False

    # feature engerneeing
    # features = ['lev_thi']  # just lev & thi columns
    features = 'all'  # all columns

    # y prediction column
    y_pred_col = ['y_eps pct']

    # window settings
    backlooking_yeras = 4

    # results location
    export_results = '/Users/vanalmsick/Workspace/MasterThesis/output/'


    ###########################################################################



    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    tracking_address = my_helpers.get_project_directories(key='tensorboard_logs')
    try:
        shutil.rmtree(tracking_address)
        time.sleep(10)
    except:
        pass
    os.mkdir(tracking_address)



    from app.b_data_cleaning import get_dataset_registry

    dataset_props = get_dataset_registry()[dataset_name]
    comp_col = dataset_props['company_col']
    time_cols = dataset_props['iter_cols']
    industry_col = dataset_props['industry_col']



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
                                    industry_col=industry_col, yearly_data=yearly_data, all_features=features)



    y_cols = ['y_eps', 'y_eps pct', 'y_dividendyield', 'y_dividendyield pct', 'y_dividendperstock', 'y_dividendperstock pct', 'y_roe', 'y_roe pct', 'y_roa', 'y_roa pct', 'y_EBIT', 'y_EBIT pct', 'y_Net income', 'y_Net income pct']



    from app.c_data_prep.ii_data_prep import data_prep

    data = data_prep(dataset=df_to_use, y_cols=y_cols, iter_cols=time_cols, comp_col=comp_col)

    qrt_multiplier = 1 if yearly_data else 4

    data.window(input_width=backlooking_yeras * qrt_multiplier, pred_width=qrt_multiplier, shift=1)

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

    #data.filter_features(just_include=['1_Inventory', '3_CAPX', '4_RnD', '5_Gross Margin', '6_Sales & Admin. Exp.', '9_Order Backlog', '10_Labor Force', '11_FIFO/LIFO dummy'])
    data.filter_y(just_include=y_pred_col)


    ############# RUN ALL MODELS ACROSS TIME #############

    run_model_acorss_time(data_obj=data, max_serach_iterations=100, MAX_EPOCHS=1000, patience=25, example_len=5, example_list=[], y_col=y_pred_col[0], export_results=export_results,
                          redo_serach_best_model=True,
                          model_name='linear_all_best',
                          activation_funcs=['linear', 'sigmoid', 'tanh', 'relu', 'elu', 'selu'],
                          NN_max_depth=4)


    ######################################################





