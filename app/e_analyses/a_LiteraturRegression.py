import os, sys, time, shutil, warnings
import mlflow
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import eli5
from eli5.sklearn import PermutationImportance

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

    from app.c_data_prep.ii_data_prep import data_prep
    data = data_prep(dataset=df_to_use, y_cols=y_cols, iter_cols=time_cols, comp_col=comp_col, keep_raw_cols=[], drop_cols=[])

    qrt_multiplier = 1 if yearly_data else 4

    data.window(input_width=5 * qrt_multiplier, pred_width=qrt_multiplier, shift=1)

    # data.block_static_split(val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=5 * qrt_multiplier * 2, val_time_steps=qrt_multiplier,
                         test_time_steps=qrt_multiplier, shuffle=True)
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
    data.filter_y(just_include=['y_eps'], exclude=None)

    out = data['200300_201900']
    train, val, test = data.df_dataset(out='all', out_dict=None)
    train_X, train_y = train
    test_X, test_y = test


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

    p_values = pd.DataFrame(index=selected_columns)
    coef = pd.DataFrame(index=selected_columns)
    r_squared = []

    # Test on all data time-steps
    for out in data:
        train, val, test = data.df_dataset(out='all', out_dict=None)
        train_X, train_y = train
        test_X, test_y = test

        model = sm.OLS(train_y.flatten(), train_X)
        results = model.fit()
        #print(results.summary())

        p_values[out['iter_step']] = results.pvalues
        coef[out['iter_step']] = results.params
        r_squared.append(results.rsquared)
        print(out['iter_step'], results.rsquared)
        print('1% level', (results.pvalues <= 0.01).value_counts()[True], '5% level', ((results.pvalues > 0.01) & (results.pvalues <= 0.05)).value_counts()[True], '10% level', ((results.pvalues > 0.05) & (results.pvalues <= 0.10)).value_counts()[True])

        import statsmodels
        ypred = results.predict(test_X)
        mae = statsmodels.tools.eval_measures.meanabs(ypred, test_y.flatten())
        print('OOS MAE:', mae)

        examples = data.get_examples(example_len=5, example_list=[])
        examples['pred'] = {}
        examples['pred']['OLS'] = np.reshape(results.predict(examples['X'][:, -1, :]), (-1, 1))

        from app.d_prediction.prediction import plot
        plot(examples_dict=examples, normalization=False)