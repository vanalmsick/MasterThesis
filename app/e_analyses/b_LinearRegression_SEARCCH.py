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


    features = ['lev_thi']
    features = 'all'
    df_to_use = feature_engerneeing(dataset=df_cleaned, comp_col=comp_col, time_cols=time_cols,
                                industry_col=industry_col, yearly_data=yearly_data, all_features=features)

    y_cols = ['y_eps', 'y_eps pct', 'y_dividendyield', 'y_dividendyield pct', 'y_dividendperstock', 'y_dividendperstock pct', 'y_roe', 'y_roe pct', 'y_roa', 'y_roa pct', 'y_EBIT', 'y_EBIT pct', 'y_Net income', 'y_Net income pct']
    df_to_use = df_to_use[df_to_use['y_eps'] < 3]


    from app.c_data_prep.ii_data_prep import data_prep
    data = data_prep(dataset=df_to_use, y_cols=y_cols, iter_cols=time_cols, comp_col=comp_col, keep_raw_cols=[], drop_cols=[])

    qrt_multiplier = 1 if yearly_data else 4

    data.window(input_width=5 * qrt_multiplier, pred_width=qrt_multiplier, shift=1)

    # data.block_static_split(val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=4 * qrt_multiplier * 2, val_time_steps=qrt_multiplier * 2,
                         test_time_steps=qrt_multiplier * 2, shuffle=True)
    # data.single_time_rolling(val_time_steps=1, test_time_steps=1, shuffle=True)

    data.normalize(method='no')
    #data.normalize(method='block')
    # data.normalize(method='time')
    # data.normalize(method='set')

    data.compute()

    print(data)

    out = data['200300_201900']
    train, val, test = data.df_dataset(out='all', out_dict=None)
    train_X, train_y = train
    test_X, test_y = test

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
    data.filter_features(just_include=columns, exclude=None)
    data.filter_y(just_include=['y_eps pct'], exclude=None)

    out = data['200300_201900']
    train, val, test = data.df_dataset(out='all', out_dict=None)
    train_X, train_y = train
    test_X, test_y = test

    """ p value Backward eliminatin
    # drop by p-value
    import statsmodels.api as sm
    def backwardElimination(x, Y, sl, columns):
        numVars = len(x[0])
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(Y, x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            if maxVar > sl:
                for j in range(0, numVars - i):
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                        x = np.delete(x, j, 1)
                        columns = np.delete(columns, j)

        regressor_OLS.summary()
        return x, columns


    SL = 0.15
    data_modeled, selected_columns = backwardElimination(train_X.values, train_y.flatten(), SL, columns)
    
    selected_columns = selected_columns.tolist()
    data.filter_features(just_include=selected_columns, exclude=None)
    data.filter_y(just_include=['y_eps'], exclude=None)

    out = data['200300_201900']
    train, val, test = data.df_dataset(out='all', out_dict=None)
    train_X, train_y = train
    test_X, test_y = test

    model = sm.OLS(train_y.flatten(), train_X)
    results = model.fit()
    print(results.summary())
    """


    print('Using these columns:', columns)
    from regressors import stats

    coll_cols=[]

    # Test on all data time-steps
    for out in [data['200000_201600'], data['200100_201700'], data['200200_201800'], data['200300_201900'], data['200400_202000']]:
        print(f"################################## {out['iter_step']} ##################################")
        train, val, test = data.df_dataset(out='all', out_dict=out)
        train_X, train_y = train

        from sklearn.preprocessing import RobustScaler
        scale_X = RobustScaler().fit(train_X)
        #scale_y = RobustScaler().fit(train_y)

        train_X = pd.DataFrame(scale_X.transform(train_X), columns=train_X.columns.tolist())
        #train_y = scale_y.transform(train_y)
        train_y = train_y.squeeze()
        val_X, val_y = val
        val_y = val_y.squeeze()
        val_X = pd.DataFrame(scale_X.transform(val_X), columns=val_X.columns.tolist())
        #val_y = scale_y.transform(val_y)
        test_X, test_y = test
        test_X = pd.DataFrame(scale_X.transform(test_X), columns=test_X.columns.tolist())
        test_y = test_y.squeeze()
        #test_y = scale_y.transform(test_y)
        oos_X = val_X.append(test_X)
        oos_y = np.append(val_y,test_y).squeeze()

        #columns_retained_RFECV = ['17_ROE', '9_inventory to assets', '5_Gross Margin_sft_4', '6_Sales & Admin. Exp._sft_4', '2_current ratio', 'y_eps', '8_inventory turnover_sft_4', 'y_dividendperstock pct', '9_Order Backlog_sft_2', '55_WC to assets_sft_4', '9_Order Backlog', '61_Repayment of LT debt ', 'y_dividendyield pct', '5_Gross Margin_sft_2', '30_sales to assets']

        #train_X = train_X[columns_retained_RFECV]
        #val_X = val_X[columns_retained_RFECV]
        #test_X = test_X[columns_retained_RFECV]
        #oos_X = oos_X[columns_retained_RFECV]




        from sklearn import linear_model
        from sklearn.feature_selection import RFECV, RFE

        # print('############ RFE ############')
        #
        # clf = linear_model.LinearRegression()
        # trans = RFE(clf, n_features_to_select=10)
        # kepler_X_trans = trans.fit_transform(train_X, train_y)
        # columns_retained_RFE = train_X.iloc[:, :].columns[trans.get_support()].values
        # print('Cols to keep:', columns_retained_RFE)
        #
        # clf = linear_model.LinearRegression().fit(train_X[columns_retained_RFE], train_y)
        # stats.summary(clf, oos_X[columns_retained_RFE], oos_y, columns_retained_RFE)
        # print('Train R:', clf.score(train_X[columns_retained_RFE], train_y))
        # print('OOS R:', clf.score(oos_X[columns_retained_RFE], oos_y))
        #
        #
        #
        #
        # print('############ RFECV ############')
        #
        # clf = linear_model.LinearRegression()
        # trans = RFECV(clf)
        # kepler_X_trans = trans.fit_transform(train_X, train_y)
        # columns_retained_RFECV = train_X.iloc[:, :].columns[trans.get_support()].values
        # print('Cols to keep:', columns_retained_RFECV)
        #
        # clf = linear_model.LinearRegression().fit(train_X[columns_retained_RFECV], train_y)
        # stats.summary(clf, oos_X[columns_retained_RFECV], oos_y, columns_retained_RFECV)
        # print('Train R:', clf.score(train_X[columns_retained_RFECV], train_y))
        # print('OOS R:', clf.score(oos_X[columns_retained_RFECV], oos_y))



        ##################################################################################
        # Construct some pipelines
        from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV #, HalvingGridSearchCV
        from sklearn.feature_selection import RFE
        from sklearn.metrics import accuracy_score, mean_absolute_error

        jobs = -1

        params = {'scaler': {'Scaler': {'with_mean': [True, False]},
                             'RobustScaler': {'with_centering': [True, False]}},
                  'regression': {'MultiTaskElasticNetCV': {'cv': [3]}},
                  'feature_selection': {'RFE': {'n_features_to_select': [7, 10, 12]}}}

        grids = {}
        for scaler_name, scaler in {'Scaler': StandardScaler, 'RobustScaler': RobustScaler}.items():
            for regression_name, regression in {'Linear': linear_model.LinearRegression, 'ElasticNetCV': linear_model.ElasticNetCV, 'MultiTaskElasticNetCV': linear_model.MultiTaskElasticNetCV}.items():
                for FS_name, FS in {'RFE': RFE}.items():
                    for fs_base_reg_name, fs_base_reg in {'Linear': linear_model.LinearRegression, regression_name: regression}.items():
                        pipe = Pipeline([(scaler_name, scaler()),
                                         (FS_name, FS(fs_base_reg())),
                                         (regression_name, regression())])

                        param_grid = {}
                        if scaler_name in params['scaler']:
                            scaler_dict = dict(zip([scaler_name + '__' + key for key in list(params['scaler'][scaler_name].keys())], list(params['scaler'][scaler_name].values())))
                            param_grid.update(scaler_dict)
                        if FS_name in params['feature_selection']:
                            feature_selection_dict = dict(zip([FS_name + '__' + key for key in list(params['feature_selection'][FS_name].keys())], list(params['feature_selection'][FS_name].values())))
                            param_grid.update(feature_selection_dict)
                        if regression_name in params['regression']:
                            regression_dict = dict(zip([regression_name + '__' + key for key in list(params['regression'][regression_name].keys())], list(params['regression'][regression_name].values())))
                            param_grid.update(regression_dict)



                        grid = GridSearchCV(estimator=pipe,
                                            param_grid=[param_grid],
                                            scoring='neg_mean_absolute_error',
                                            cv=3,
                                            n_jobs=jobs)

                        grids[scaler_name + '_' + FS_name + f'({fs_base_reg_name})' + '_' + regression_name] = grid



        warnings.filterwarnings("ignore")

        # Fit the grid search objects
        print('Performing model optimizations...')
        best_acc = 0.0
        best_clf = 0
        best_gs = ''
        best_cols = []
        for grid_name, grid in grids.items():
            print('\nEstimator: %s' % grid_name)
            # Fit grid search
            grid.fit(train_X, train_y.reshape((-1,1)))
            # Best params
            print('Best params: %s' % grid.best_params_)
            # Cols
            relevant_cols = train_X.iloc[:, :].columns[grid.best_estimator_.steps[1][1].get_support()].tolist()
            print('Columns used:', relevant_cols)
            # Best training data accuracy
            print('Best training accuracy: %.3f' % grid.best_score_)
            # Predict on test data with best params
            y_pred = grid.predict(oos_X)
            # Test data accuracy of model with best params
            error = -mean_absolute_error(oos_y, y_pred.reshape((-1,1)))
            print('Test set accuracy score for best params: %.3f ' % error)
            # Track best (highest test accuracy) model
            if error > best_acc:
                best_acc = error
                best_gs = grid
                best_clf = grid_name
                best_cols = relevant_cols

        print('\nClassifier with best test set accuracy: %s' % best_clf)
        with open('best_linear_model.txt', 'a') as f:
            f.write(f'Best model: {best_clf}\n')
            f.write(f'Error: {best_acc}\n')
            f.write(f'Params: {best_gs}\n')
            f.write(f'Cols: {best_cols}\n')
            f.write(f'--------------------------\n')





