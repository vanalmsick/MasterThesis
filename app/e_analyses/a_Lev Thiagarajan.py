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
    df_to_use = feature_engerneeing(dataset=df_cleaned, comp_col=comp_col, time_cols=time_cols,
                                industry_col=industry_col, yearly_data=yearly_data, all_features=features)

    y_cols = ['y_eps', 'y_eps pct', 'y_dividendyield', 'y_dividendyield pct', 'y_dividendperstock', 'y_dividendperstock pct', 'y_roe', 'y_roe pct', 'y_roa', 'y_roa pct', 'y_EBIT', 'y_EBIT pct', 'y_Net income', 'y_Net income pct']

    from app.c_data_prep.ii_data_prep import data_prep
    data = data_prep(dataset=df_to_use, y_cols=y_cols, iter_cols=time_cols, comp_col=comp_col, keep_raw_cols=[],
                 drop_cols=[])

    qrt_multiplier = 1 if yearly_data else 4

    data.window(input_width=5 * qrt_multiplier, pred_width=qrt_multiplier, shift=1)

    # data.block_static_split(val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=5 * qrt_multiplier * 2, val_time_steps=qrt_multiplier,
                         test_time_steps=qrt_multiplier, shuffle=True)
    # data.single_time_rolling(val_time_steps=1, test_time_steps=1, shuffle=True)

    # data.normalize(method='block')
    data.normalize(method='time')
    # data.normalize(method='set')

    data.compute()

    print(data)


    # Filter
    data.filter_features(just_include=['1_inventory', '3_cap exp', '4_RnD', '5_gross margin', '6_sales admin exp', '9_order backlog', '10_labor force', '11_FIFO dummy', '11_LIFO dummy'], exclude=None)
    data.filter_y(just_include=['y_eps'], exclude=None)
    data.filter_companies(just_include=None, exclude=None)


    out = data['200300_201900']
    data_props = data.get_data_props()

    train_np, val_np, test_np = data.df_dataset(out='all', out_dict=None)
    examples = data.get_examples(example_len=5, example_list=[])
    examples['pred'] = {}



    train_X, train_y = train_np
    test_X, test_y = test_np
    #test_X = sm.add_constant(test_X, has_constant='add')
    #train_X = sm.add_constant(train_X, has_constant='add')

    logisticRegr = LinearRegression()
    logisticRegr.fit(train_X, train_y)

    model = sm.OLS(train_y.flatten(), train_X)
    results = model.fit()
    print(results.summary())

    #perm = PermutationImportance(logisticRegr, random_state=1).fit(test_X, test_y)
    #feature_importances = pd.DataFrame(np.array([['const'] + data_props['final_data']['cols']['X'], perm.feature_importances_, perm.feature_importances_std_]).T, columns=['feature', 'feature_importance', 'feature_importance_std'])
    #feature_importances['feature_importance_abs'] = abs(feature_importances['feature_importance'].astype(float))
    #feature_importances.sort_values('feature_importance_abs', ascending=False, inplace=True)
    #print(feature_importances)

    import numpy as np
    from scipy.stats import norm
    from sklearn.linear_model import LogisticRegression
    import sklearn


    def logit_pvalue(model, x, round=False):
        """ Calculate z-scores for scikit-learn LogisticRegression.
        parameters:
            model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
            x:     matrix on which the model was fit
        This function uses asymtptics for maximum likelihood estimates.
        """
        p = model.predict_proba(x)
        n = len(p)
        m = len(model.coef_[0]) + 1
        coefs = np.concatenate([model.intercept_, model.coef_[0]])
        x_full = np.matrix(np.insert(np.array(x), 0, 1, axis=1))
        ans = np.zeros((m, m))
        for i in range(n):
            ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i, 1] * p[i, 0]
        try:
            vcov = np.linalg.inv(np.matrix(ans))
        except:
            warnings.warn('Use pseudo inverse matrix because of singular matrix issue.')
            vcov = np.linalg.pinv(np.matrix(ans))
        se = np.sqrt(np.diag(vcov))
        t = coefs / se
        p = (1 - norm.cdf(abs(t))) * 2
        if round != False:
            p = p.round(round)
        return p


    #print(logit_pvalue(logisticRegr, train_X))


