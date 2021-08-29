import os

import mlflow.keras

# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')
from app.z_helpers import helpers as my_helpers

from app.d_prediction.a_tf_base import run_model_acorss_time






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
    features = ['lev_thi']  # just lev & thi columns
    # features = 'all'  # all columns

    # y prediction column
    y_pred_col = ['y_eps pct']

    # window settings
    backlooking_yeras = 4

    # results location
    export_results = False
    export_results = '/Users/vanalmsick/Workspace/MasterThesis/results/'

    model_name = 'dense_lit_best'

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

    data.normalize(method='no')
    #data.normalize(method='block')
    #data.normalize(method='time')
    # data.normalize(method='set')

    data.compute()
    print(data)

    data.filter_features(just_include=['1_Inventory', '3_CAPX', '4_RnD', '5_Gross Margin', '6_Sales & Admin. Exp.', '9_Order Backlog', '10_Labor Force', '11_FIFO/LIFO dummy'] + y_pred_col)
    data.filter_y(just_include=y_pred_col)


    ############# RUN ALL MODELS ACROSS TIME #############

    run_model_acorss_time(data_obj=data, max_serach_iterations=250, MAX_EPOCHS=1000, patience=25, example_len=5, example_list=[], y_col=y_pred_col[0], export_results=export_results,
                          redo_serach_best_model=True,
                          model_name=model_name,
                          activation_funcs=['linear', 'sigmoid', 'tanh', 'relu', 'elu', 'selu'],
                          NN_max_depth=10)


    ######################################################





