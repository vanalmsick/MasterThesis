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
    # MLFLOW_TRACKING_URI = 'file:' + '/Users/vanalmsick/Workspace/MasterThesis/cache/MLflow'  # with local tracking serveer
    MLFLOW_TRACKING_URI = 'http://0.0.0.0:5000'  # with remote tracking server with registry
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
    all_cols = [comp_col] + time_cols + y_col + ['1_Inventory', '3_CAPX', '4_RnD', '5_Gross Margin', '6_Sales & Admin. Exp.', '9_Order Backlog', '10_Labor Force', '11_FIFO/LIFO dummy']

    df_to_use = df_to_use[all_cols]

    df_to_use = df_to_use[(df_to_use[all_cols[4:-1]] < df_to_use[all_cols[4:-1]].quantile(.975)).all(axis=1)]
    df_to_use = df_to_use[(df_to_use[all_cols[4:-1]] > df_to_use[all_cols[4:-1]].quantile(.025)).all(axis=1)]
    print('90-quantil brder:', len(df_to_use))


    from app.c_data_prep.ii_data_prep import data_prep
    data = data_prep(dataset=df_to_use, y_cols=y_col, iter_cols=time_cols, comp_col=comp_col, keep_raw_cols=[], drop_cols=[])

    qrt_multiplier = 1 if yearly_data else 4

    data.window(input_width=1 * qrt_multiplier, pred_width=qrt_multiplier, shift=1)

    # data.block_static_split(val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=6 * qrt_multiplier * 2, val_time_steps=qrt_multiplier * 2,
                         test_time_steps=qrt_multiplier * 2, shuffle=True)
    # data.single_time_rolling(val_time_steps=1, test_time_steps=1, shuffle=True)

    data.normalize(method='no')
    #data.normalize(method='block')
    # data.normalize(method='time')
    # data.normalize(method='set')

    data.compute()

    print(data)

    y_col = ['y_eps pct']
    X_cols = y_col + ['1_Inventory', '3_CAPX', '4_RnD', '5_Gross Margin', '6_Sales & Admin. Exp.', '9_Order Backlog', '10_Labor Force', '11_FIFO/LIFO dummy']
    data.filter_features(just_include=X_cols)
    data.filter_y(just_include=y_col)

    p_value_summary = pd.DataFrame(index=['Regression Model'] + ['_intercept'] + X_cols)
    best_model_coef_summary = pd.DataFrame(index=['Regression Model'] + ['_intercept'] + X_cols)


    # Test on all data time-steps
    for out in [data['200000_201600'], data['200100_201700'], data['200200_201800'], data['200300_201900']]: #, data['200400_202000']
        print(out['iter_step'])
        train, val, test = data.df_dataset(out='all', out_dict=out)
        train_X, train_y = train

        from sklearn.preprocessing import RobustScaler, StandardScaler  # RobustScaler worked by far better
        from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, BayesianRidge



        for regression_name, regression in zip(['LinearRegression', 'RidgeCV', 'LassoCV', 'ElasticNetCV', 'BayesianRidge'], [LinearRegression(), RidgeCV(cv=3), LassoCV(cv=3), ElasticNetCV(cv=3), BayesianRidge()]):

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

            from sklearn.feature_selection import RFECV, RFE
            from sklearn import linear_model
            import warnings
            warnings.simplefilter("ignore")


            from sklearn.linear_model import ElasticNetCV


            #regr = ElasticNetCV(cv=5, random_state=0)
            regr = regression
            regr.fit(train_X, train_y)
            print(f'--------------------------------- {regression_name} ---------------------------------')
            print('Train:', regr.score(train_X, train_y))
            print('Val:', regr.score(val_X, val_y))
            print('OOS:', regr.score(oos_X, oos_y))


            from regressors import stats
            #print(stats.summary(regr, train_X, train_y.squeeze(), train_X.columns.tolist()))
            print(stats.summary(regr, oos_X, oos_y.squeeze().reshape(-1, 1), oos_X.columns.tolist()))
            p_vals = pd.Series(stats.coef_pval(regr, oos_X, oos_y.squeeze().reshape(-1, 1)),index=['_intercept'] + X_cols, name=out['iter_step'])
            p_vals['Regression Model'] = regression_name
            p_value_summary = pd.concat((p_value_summary, p_vals), axis=1)

            best_model = 'LinearRegression'
            if regression_name == best_model:
                coef = np.concatenate((np.round(np.array([regr.intercept_]).reshape(-1, 1), 6), np.round((regr.coef_.reshape(-1, 1)), 6))).squeeze()
                coef = pd.Series(coef, index=['_intercept'] + X_cols, name=out['iter_step'])
                coef['Regression Model'] = regression_name
                best_model_coef_summary = pd.concat((best_model_coef_summary, coef), axis=1)

        my.word_regression_table(model=regr, X=train_X, y=pd.Series(train_y.squeeze(), name='y_eps'), output_file=f"/Users/vanalmsick/Workspace/MasterThesis/output/regression_lin_LevThi-{out['iter_step']}.docx", title=None)


        #examples = data.get_examples(example_len=5, example_list=[])
        #examples['pred'] = {}
        #examples['pred']['OLS'] = np.reshape(results.predict(examples['X'][:, -1, :]), (-1, 1))

        from app.d_prediction.prediction import plot
        #plot(examples_dict=examples, normalization=False)

    p_value_summary.columns = pd.MultiIndex.from_product([["Regression Model"], p_value_summary.columns])
    with pd.ExcelWriter('/Users/vanalmsick/Workspace/MasterThesis/output/regression_lin_LevThi-compare.xlsx') as writer:
        p_value_summary.to_excel(writer, sheet_name='P Values - All models')
        best_model_coef_summary.to_excel(writer, sheet_name=f'Coef Values - {best_model}')
