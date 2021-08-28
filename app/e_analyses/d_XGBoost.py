import os, sys, time, shutil, warnings
import mlflow
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
#import eli5
#from eli5.sklearn import PermutationImportance

if __name__ == '__main__':  # must be in if condition because I am pusing parallel processing
    # Working directory must be the higher .../app folder
    from app.z_helpers import helpers as my
    my.convenience_settings()

    ### run tensorboard
    MLFLOW_TRACKING_URI = 'file:' + '/Users/vanalmsick/Workspace/MasterThesis/cache/MLflow'  # with local tracking serveer
    #MLFLOW_TRACKING_URI = 'http://0.0.0.0:5000'  # with remote tracking server with registry
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)




    # Dataset to use
    dataset_name = 'handpicked_dataset'
    yearly_data = True

    from app.b_data_cleaning import get_dataset_registry

    dataset_props = get_dataset_registry()[dataset_name]
    comp_col = dataset_props['company_col']
    time_cols = dataset_props['iter_cols']
    industry_col = dataset_props['industry_col']

    recache_raw_data = False
    redo_data_cleaning = False

    # Cleaining settings
    required_filled_cols_before_filling = ['sales', 'roe', 'ebit']
    required_filled_cols_after_filling = []
    drop_threshold_row_pct = 0.4
    drop_threshold_row_quantile = 0.15
    drop_threshold_col_pct = 0
    append_data_quality_col = False

    from app.c_data_prep.i_feature_engineering import get_clean_data, feature_engerneeing

    df_cleaned = get_clean_data(dataset_name, recache_raw_data=recache_raw_data, redo_data_cleaning=redo_data_cleaning,
                            comp_col=comp_col, time_cols=time_cols, industry_col=industry_col,
                            required_filled_cols_before_filling=required_filled_cols_before_filling,
                            required_filled_cols_after_filling=required_filled_cols_after_filling,
                            drop_threshold_row_pct=drop_threshold_row_pct,
                            drop_threshold_row_quantile=drop_threshold_row_quantile,
                            drop_threshold_col_pct=drop_threshold_col_pct,
                            append_data_quality_col=append_data_quality_col)
    print('Data cleaned:', len(df_cleaned))

    features = ['lev_thi']

    df_to_use = feature_engerneeing(dataset=df_cleaned, comp_col=comp_col, time_cols=time_cols,
                                industry_col=industry_col, yearly_data=yearly_data, all_features=features)
    print('Data to use:', len(df_to_use))

    #y_cols = ['y_eps', 'y_eps pct', 'y_dividendyield', 'y_dividendyield pct', 'y_dividendperstock', 'y_dividendperstock pct', 'y_roe', 'y_roe pct', 'y_roa', 'y_roa pct', 'y_EBIT', 'y_EBIT pct', 'y_Net income', 'y_Net income pct']
    df_to_use = df_to_use[abs(df_to_use['y_eps pct']) < 200]
    print('200% brder:', len(df_to_use))

    y_col = ['y_eps pct']
    # all_cols = [comp_col] + time_cols + y_col + ['1_Inventory', '3_CAPX', '4_RnD', '5_Gross Margin', '6_Sales & Admin. Exp.', '9_Order Backlog', '10_Labor Force', '11_FIFO/LIFO dummy']
    #
    # df_to_use = df_to_use[all_cols]
    #
    # df_to_use = df_to_use[(df_to_use[all_cols[4:-1]] < df_to_use[all_cols[4:-1]].quantile(.975)).all(axis=1)]
    # df_to_use = df_to_use[(df_to_use[all_cols[4:-1]] > df_to_use[all_cols[4:-1]].quantile(.025)).all(axis=1)]
    print('90-quantil brder:', len(df_to_use))


    from app.c_data_prep.ii_data_prep import data_prep
    data = data_prep(dataset=df_to_use, y_cols=y_col, iter_cols=time_cols, comp_col=comp_col, keep_raw_cols=[], drop_cols=[])

    qrt_multiplier = 1 if yearly_data else 4

    data.window(input_width=4 * qrt_multiplier, pred_width=qrt_multiplier, shift=1)

    # data.block_static_split(val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0,
                             train_time_steps=5 * qrt_multiplier * 2,
                             val_time_steps=qrt_multiplier * 2,
                             test_time_steps=qrt_multiplier * 2, shuffle=True)
    # data.single_time_rolling(val_time_steps=1, test_time_steps=1, shuffle=True)

    data.normalize(method='no')
    #data.normalize(method='block')
    # data.normalize(method='time')
    # data.normalize(method='set')

    data.compute()

    print(data)

    y_col = ['y_eps pct']
    #X_cols = y_col + ['1_Inventory', '3_CAPX', '4_RnD', '5_Gross Margin', '6_Sales & Admin. Exp.', '9_Order Backlog', '10_Labor Force', '11_FIFO/LIFO dummy']
    #data.filter_features(just_include=X_cols)
    data.filter_y(just_include=y_col)


    # Test on all data time-steps
    for out in [data['200000_201700'], data['200100_201800'], data['200200_201900'], data['200300_202000']]: #, data['200400_202000']
        print(out['iter_step'])
        train, val, test = data.df_dataset(out='all', out_dict=out)
        train_X, train_y = train

        from sklearn.preprocessing import RobustScaler

        transformer_X = RobustScaler().fit(train_X)
        train_X = pd.DataFrame(transformer_X.transform(train_X), columns=train_X.columns)

        transformer_y = RobustScaler().fit(train_y)
        train_y = transformer_y.transform(train_y)

        val_X, val_y = val
        test_X, test_y = test
        oos_X = val_X.append(test_X)
        oos_X = pd.DataFrame(transformer_X.transform(oos_X), columns=oos_X.columns)
        oos_y = np.append(val_y, test_y).reshape(-1, 1)
        oos_y = transformer_y.transform(oos_y)

        print('Train size:', len(train_X))
        print('Val size:', len(val_X))
        print('Test size:', len(test_X))
        print('OOS size:', len(oos_X))

        from xgboost import XGBRegressor
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.model_selection import KFold
        from sklearn.model_selection import cross_val_score

        model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        #model.fit(train_X, train_y)
        #kfold = KFold(n_splits=5) #, random_state=42
        #results = cross_val_score(model, train_X, train_y, cv=kfold)

        # Create parameter grid
        parameters = {"learning_rate": [0.1, 0.01, 0.001],
                      "gamma": [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
                      "max_depth": [2, 4, 7, 10],
                      "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
                      "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
                      "reg_alpha": [0, 0.5, 1],
                      "reg_lambda": [1, 1.5, 2, 3, 4.5],
                      "min_child_weight": [1, 3, 5, 7],
                      "n_estimators": [100, 250, 500, 1000]}

        # Create RandomizedSearchCV Object
        xgb_rscv = RandomizedSearchCV(model, param_distributions=parameters, #scoring="f1_micro",
                                      cv=7, verbose=3, random_state=40)

        # Fit the model
        model_xgboost = xgb_rscv.fit(train_X, train_y, eval_metric="mae", eval_set=[(val_X, val_y)], verbose=False)

        print('===================================================================')

        print("Learning Rate: ", model_xgboost.best_estimator_.get_params()["learning_rate"])
        print("Gamma: ", model_xgboost.best_estimator_.get_params()["gamma"])
        print("Max Depth: ", model_xgboost.best_estimator_.get_params()["max_depth"])
        print("Subsample: ", model_xgboost.best_estimator_.get_params()["subsample"])
        print("Max Features at Split: ", model_xgboost.best_estimator_.get_params()["colsample_bytree"])
        print("Alpha: ", model_xgboost.best_estimator_.get_params()["reg_alpha"])
        print("Lamda: ", model_xgboost.best_estimator_.get_params()["reg_lambda"])
        print("Minimum Sum of the Instance Weight Hessian to Make a Child: ", model_xgboost.best_estimator_.get_params()["min_child_weight"])
        print("Number of Trees: ", model_xgboost.best_estimator_.get_params()["n_estimators"])

        print("\nTrain Score:", model_xgboost.score(train_X, train_y))
        print("\nOOS Score:", model_xgboost.score(oos_X, oos_y))

        print('===================================================================')


        examples = data.get_examples(example_len=5, example_list=[])
        examples['pred'] = {}
        preds = []
        for example in examples['X']:
            preds.append(float(model_xgboost.predict(example)))
        examples['pred']['XGBoost'] = np.reshape(preds, (-1, 1))

        from app.d_prediction.prediction import plot
        plot(examples_dict=examples, normalization=False)


