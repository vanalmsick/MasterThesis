import os, sys
import pickle, warnings
import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.linear_model import LinearRegression

# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')
from app.z_helpers import helpers as my



def _download_data_from_sql(data_version='final_data', recache=False):
    from app.b_data_cleaning import get_dataset_registry
    sql_table_name = get_dataset_registry()[data_version]['sql_table']
    query = "SELECT * FROM {}".format(sql_table_name)

    param_dic = my.get_credentials(credential='aws')

    cache_folder = os.path.join(my.get_project_directories(key='cache_dir'), 'raw_data')
    data_file = os.path.join(cache_folder, (data_version + '.csv'))
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    if recache or not os.path.exists(data_file):
        print('Getting raw data via sql...')

        with my.postgresql_connect(param_dic) as conn:
            df = pd.read_sql_query(query, con=conn)
            obj_cols = df.select_dtypes(include='object').columns
            df[obj_cols] = df[obj_cols].astype(str)
            df.to_csv(data_file, index=False)
            with open(data_file[:-4] + '.dtypes', 'wb') as f:
                dtypes = df.dtypes.to_dict()
                dtypes = dict(zip(dtypes.keys(), [str if i == np.object else i for i in dtypes.values()]))
                pickle.dump(dtypes, f)
        print('Raw data cached.')

    else:
        print('Raw data already cached.')
        with open(data_file[:-4] + '.dtypes', 'rb') as f:
            dtypes = pickle.load(f)

        df = pd.read_csv(data_file, dtype=dtypes, index_col=False)

    if data_version == 'handpicked_dataset':
        app_dir = my.get_project_directories(key='app_dir')
        file_path = os.path.join(app_dir, 'a_get_data', 'reuters_eikon', 'key_reuters_fields.csv')
        data_dict = pd.read_csv(file_path)
        data_dict['Clear Name'] = data_dict['Clear Name'].str.lower()
        data_dict = data_dict.set_index('Clear Name')
        new_data_dict = data_dict[['Data Type', 'Variable Type']].to_dict(orient='index')

        fillnan_cols = []
        formula_methods = []
        for col in data_dict.columns.tolist():
            if col[:8] == 'fillnan_':
                fillnan_cols.append(col)
        fillnan_cols = sorted(fillnan_cols, key=str.lower)

        for index, row in data_dict[fillnan_cols].iterrows():
            tmp = row.tolist()
            tmp = [x for x in tmp if str(x) != 'nan']
            new_data_dict[index]['Fill NaN Rules'] = tmp
            for j in [i.split(':')[1] for i in tmp if i.split(':')[0] == 'formula']:
                formula_methods.append((index, j))

    else:
        new_data_dict = None
        formula_methods = None


    return df, data_file, new_data_dict, formula_methods


def _shift_array(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def _join_x_and_y(x, y, drop_nan=True):
    out = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1))))
    if drop_nan:
        out = out[~np.isnan(out).any(axis=1), :]
    return out


def _custom_score(y_true, y_pred):
    i = len(y_true)
    i_th = (2/3) / (i - 1)
    x = abs(y_pred - y_true) / abs(y_true)
    i_arr = np.array([1 - j * i_th for j in range(i)])
    return sum(x * i_arr) / sum(i_arr)


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]



def _arima_fill(historic, prediction_intervals=4, return_conf_int=False):
    historic = np.asarray(historic)
    if historic.ndim != 1:
        raise Exception(f'historic has to be 1d and has {historic.ndim} dimensions')

    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
    from statsmodels.tsa.arima.model import ARIMA
    from pandas.plotting import autocorrelation_plot
    from statsmodels.graphics.tsaplots import plot_pacf
    import matplotlib.pyplot as plt



    params2 = {'start_p':1, 'start_q':1,
               'test':'adf',
               'max_p': 4, 'max_q': 4, 'm': 1,
               'start_P': 0, 'seasonal': False,
               'd': None, 'D': 1, 'trace': True,
               'error_action': 'ignore',
               'suppress_warnings': True,
               'stepwise': True}

    arima_model = pm.auto_arima(historic, **params2)
    preds, conf_int = arima_model.predict(n_periods=prediction_intervals, return_conf_int=True)

    if return_conf_int:
        return preds, conf_int
    else:
        return preds


def smooth(x,window_len=11,window='hanning'):
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y




def fill_analy_est(df, base_col, est_cols):
    df = df[[base_col] + est_cols].sort_index()
    pred_periods = len(est_cols)

    for index, row in df.iterrows():
        if row.isnull().any():
            historic = df.loc[:index][base_col].values
            nan_est = _arima_fill(historic=historic, prediction_intervals=pred_periods, return_conf_int=False)
            print(nan_est)





class dataset_nan_fill:
    def __init__(self, df, company_col, time_cols, industry_col, data_props, fillnan_formulas, formula_iterations=3):

        ric_industry_mapping = df[['ric', 'industry']].set_index('ric').to_dict()['industry']
        fill_mean_df = self._get_industry_avg(df=df, industry_col=industry_col, time_cols=time_cols)

        df.set_index([company_col] + time_cols, inplace=True)
        df.sort_index(inplace=True)
        df.to_csv('before_nan_filling.csv', index=True)
        # Iterate through first level of dataframe index
        #df = df.loc[('MEI.N')]
        for ric, partial_df in df.groupby(level=0):
            print('ric', ric)

            # Try to fill NaNs first with formulas formula_iterations times then use other rules
            for _ in range(formula_iterations):  # try formula_iterations times to fill nen with formulas
                for col, formula in fillnan_formulas:
                    method = ['formula:' + formula]
                    partial_arr = partial_df[col].values
                    if np.isnan(partial_arr).any():
                        partial_arr = self._all_in_one(arr=partial_arr, partial_df=partial_df, col=col, methods=method, mean_df=None)
                        partial_df[col] = partial_arr


            # Use all rules (not just formula rules) - iterate across columns
            for _ in range(2):  # Do it twice so that maybe remaining NaNs depening on other fills that are not formulas but values or methods can be filled
                for col, props in data_props.items():
                    methods = props['Fill NaN Rules']
                    if len(methods) > 0 and col in df:
                        partial_arr = partial_df[col].values
                        # Check for nan/missing values
                        if np.isnan(partial_arr).any():
                            entire_mean_df = None
                            if 'approx' in str(methods):
                                tmp_approx_df = df[df[industry_col] == ric_industry_mapping[ric]].copy()
                                tmp_approx_df = tmp_approx_df.reset_index().set_index(time_cols)
                                tmp_approx_df.sort_index(inplace=True)
                                entire_mean_df = tmp_approx_df.loc[slice(partial_df.index[0][1:], partial_df.index[-1][1:])]
                            #if len(partial_df) == 1:
                            #    idx = partial_df.index[0][1:]
                            #else:
                            idx = slice(partial_df.index[0][1:], partial_df.index[-1][1:])
                            tmp_mean = fill_mean_df.loc[(ric_industry_mapping[ric])].loc[idx][col].values
                            partial_arr = self._all_in_one(arr=partial_arr, partial_df=partial_df, col=col, methods=methods, mean_df=tmp_mean, entire_mean_df=entire_mean_df)
                            partial_df[col] = partial_arr
                df.loc[(ric)] = partial_df
        df.to_csv('after_nan_filling.csv', index=True)
        print(df)



    def _get_industry_avg(self, df, industry_col, time_cols):
        df = df.copy()

        # drop wrong rows
        df = df[df['data_qrt'] != 0]
        df = df[df['data_qrt'] <= 4]

        df.set_index([industry_col] + time_cols, inplace=True)
        df = df.sort_index()


        drop_cols = ['data_date', 'data_month', 'data_day', 'year_qrt_id']
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

        all_count = df.groupby(level=[1, 2]).count()

        col_count = all_count.median(axis=0)
        #additional_drop_cols = col_count[col_count < 20].index.tolist()
        #print('Also dropping these cols because too little filled:', additional_drop_cols)
        #drop_cols = drop_cols + additional_drop_cols
        #df.drop(columns=drop_cols, inplace=True, errors='ignore')


        all_mean = df.groupby(level=[1, 2]).mean()
        all_std = df.groupby(level=[1, 2]).std()
        all_count = df.groupby(level=[1, 2]).count()
        all_mean_pct = all_mean.pct_change(periods=1)
        all_std_div_mean = all_std/all_mean

        industry_mean = df.groupby(level=[0, 1, 2]).mean()
        industry_std = df.groupby(level=[0, 1, 2]).std()
        industry_count = df.groupby(level=[0, 1, 2]).count()
        industry_mean_pct = industry_mean.pct_change(periods=1)

        all_std_div_mean = all_std_div_mean.reindex(industry_mean.reorder_levels((1, 2, 0)).index, method='ffill').reorder_levels((2, 0, 1))  # expand third level industry
        all_mean_exp = all_mean.reindex(industry_mean.reorder_levels((1, 2, 0)).index, method='ffill').reorder_levels( (2, 0, 1))  # expand third level industry

        score_part1 = (industry_count / 10) * industry_count.div(industry_count['ric'], axis=0)
        score_part1[score_part1 > 1] = 1  # max 100%
        score_part2 = 1 - (1 /( 1 + np.exp(-(industry_std / industry_mean) / all_std_div_mean + 1)))
        score = (1/3) * score_part1 + (2/3) * score_part2
        score.to_csv('score.csv')

        fill_nan_df = industry_mean * score + all_mean_exp * (1 - score)
        #print(fill_nan_df)

        # Fill with industry mean or all mean in case both is not avalable
        fill_nan_df[fill_nan_df.isnull()] = industry_mean
        fill_nan_df[fill_nan_df.isnull()] = all_mean_exp

        return fill_nan_df





    def _all_in_one(self, arr, partial_df, col, methods, mean_df=None, entire_mean_df=None):
        """
        >>> self = dataset_nan_fill(df=pd.DataFrame())
        >>> import numpy as np
        >>> a = np.array([np.nan, np.nan, np.nan, np.nan, np.nan,  49.,  49.,  49., 49.,  45.,  45.,  45., 45., np.nan, np.nan, np.nan, np.nan, 108., 108.])
        >>> self._all_in_one(arr=a, methods=['linear'])
        array([ 49. ,  49. ,  49. ,  49. ,  49. ,  49. ,  49. ,  49. ,  49. ,
                45. ,  45. ,  45. ,  45. ,  76.5,  76.5,  76.5,  76.5, 108. ,
               108. ])
        """
        # Tranform list of method strings to list of dictionaries
        transformed_methods = []
        for method in methods:
            tmp = {}
            mod_method = method.replace('(',':')
            tmp['type'] = mod_method.split(':')[0]
            tmp['formula'] = mod_method.split(':')[1]
            if '(' in method:
                kwargs = method.split('(')[1][:-1]
                kwargs = kwargs.replace('  ', ' ').replace(' ,', ',').replace(', ', ',').split(',')
                new_kwargs = {}
                for i in kwargs:
                    try:
                        new_kwargs[i.split('=')[0]] = i.split('=')[1]
                        tmp['kwargs'] = new_kwargs
                    except:
                        warnings.warn(f"NaN filling kwargs '{i}' are invalid because missing '=' sign!")

            else:
                tmp['kwargs'] = {}
            transformed_methods.append(tmp)
        methods = transformed_methods


        # Check if data quaterly and transform array if so
        qrt_data = self._check_quaterly(arr=arr)
        tmp_arr = arr
        if qrt_data:
            tmp_arr, idx, idx_mapping = self._quaterly_idxs(arr=arr)
            if not np.isnan(tmp_arr).any():  # if nans disappear when transforming it to qrt-data handle normal
                tmp_arr = arr
                qrt_data = False



        # Iterate through methods to fill nan
        for method in methods:  # Loop through list of methods to perform to fill NaNs
            if np.isnan(arr).any():  # Check if still NaNs left
                fill_values = []
                if method['type'] == 'method':
                    nan_todos = self._get_nan_blocks(arr=tmp_arr)
                    for todo_dict in nan_todos:
                        if method['formula'] == 'linear_or_mean' or method['formula'] == 'linear' or method['formula'] == 'linear_or_0':
                            if 'or' in method['kwargs']:
                                if method['kwargs']['or'].lower() == 'nan':
                                    mean_df = np.nan
                                else:
                                    mean_df = float(method['kwargs']['or'])
                            elif method['formula'] == 'linear_or_0':
                                mean_df = 0
                            tmp_fill_values = self._fill_linear(arr=todo_dict['arr'], case=todo_dict['case'], x=todo_dict['idx'][0], y=todo_dict['idx'][1], offset=todo_dict['offset'], mean_df=mean_df)
                            fill_values.extend(tmp_fill_values)
                        elif method['formula'] == 'avg':
                            # ToDo: columnname & periods as kwargs
                            pass

                        elif method['formula'] == 'approx':
                            # do later
                            pass
                        else:
                            warnings.warn(f"Unknown Fill NaN method formula '{method['formula']}'")

                    if qrt_data and method['formula'] != 'approx':
                        new_fill_values = []
                        for i, value in fill_values:
                            for j in range(idx_mapping[i][0], idx_mapping[i][1]):
                                new_fill_values.append((j, value))
                        fill_values = new_fill_values

                    if method['formula'] == 'approx':
                        other_cols = method['kwargs']['other']
                        if other_cols.find('[') != -1:
                            other_cols = other_cols.strip('][').split(', ')
                        else:
                            other_cols = [other_cols]
                        other_cols = [i.lower() for i in other_cols]
                        tmp_fill_values = self._fill_linear_approx_column(arr=todo_dict['arr'], partial_df=partial_df, col=col, other_cols=other_cols, x=todo_dict['idx'][0], y=todo_dict['idx'][1], offset=todo_dict['offset'], entire_mean_df=entire_mean_df)
                        fill_values.extend(tmp_fill_values)
                        if len(arr) != len(partial_df):
                            print('ERROR')


                elif method['type'] == 'value' or method['type'] == 'number':
                    number = float(method['formula'])
                    where_are_NaNs = np.isnan(arr)
                    arr[where_are_NaNs] = number


                elif method['type'] == 'formula':
                    fill_values = self._fill_formula(partial_df=partial_df, col=col, formula=method['formula'])

                else:
                    warnings.warn(f"Unknown Fill NaN method type '{method['type']}'")


                for i, value in fill_values:
                    if np.isnan(arr[i]):
                        if value not in [np.nan, np.inf, -np.inf]:
                            arr[i] = value

        return arr


    def _quaterly_idxs(self, arr):
        last_value = arr[0]
        for i in range(len(arr)):
            value = arr[i]
            if abs(last_value - value) > 0.001 or (np.isnan(last_value) and not np.isnan(value)) or (np.isnan(value) and not np.isnan(last_value)):
                if i >= 4:
                    final_i = i % 4
                else:
                    final_i = i
                break
            last_value = value

        i_list = [0] + list(range(final_i, len(arr), 4)) if final_i != 0 else list(range(final_i, len(arr), 4))
        final_arr = arr[i_list]
        i_mapping = dict(zip(range(len(i_list)), [(i_list[i], ((i_list[1:] + [len(arr)])[i])) for i in range(len(i_list))]))
        return final_arr, i_list, i_mapping






################


    def _fill_formula(self, partial_df, col, formula):
        signs = [i for i in formula if i in ['/', '*', '+', '-']]

        # Get sign positions in string formula
        sign_positions = []
        last_i = -1
        for sign in signs:
            last_i = formula.find(sign, last_i + 1)
            sign_positions.append(last_i)

        # Get columns
        cols = formula.replace('/', '@').replace('*', '@').replace('+', '@').replace('-', '@').lower()
        cols = cols.split('@')

        # Fill nan
        nan_df = partial_df[partial_df[col].isnull()]
        nan_rows = [j for j, i in zip(range(0, len(partial_df)), partial_df[col].isnull()) if i]
        if cols[0] == '':
            tmp = 0
        else:
            if cols[0].isnumeric():
                tmp = float(cols[0])
            else:
                tmp = nan_df[cols[0]]
        for i in range(len(signs)):  # For columns
            sign = signs[i]
            sign_col = cols[i + 1]
            if sign_col in nan_df:
                if sign == '*':
                    tmp = tmp * nan_df[sign_col]
                elif sign == '/':
                    tmp = tmp / nan_df[sign_col]
                elif sign == '+':
                    tmp = tmp + nan_df[sign_col]
                elif sign == '-':
                    tmp = tmp - nan_df[sign_col]
            elif sign_col.isnumeric():  # For values
                sign_col = float(sign_col)
                if sign == '*':
                    tmp = tmp * sign_col
                elif sign == '/':
                    tmp = tmp / sign_col
                elif sign == '+':
                    tmp = tmp + sign_col
                elif sign == '-':
                    tmp = tmp - sign_col
            else:
                warnings.warn(f'Column {sign_col} not in DataFrame. Could not fill column {col} accoring to rule: {formula}. Please correct wrong column name in formula!')

        # Tranform to fill nan values format
        new_nan = []
        for i in range(len(nan_df)):
            new_nan.append((nan_rows[i], tmp.iloc[i]))

        return new_nan



    def _fill_linear(self, arr, case, x, y, offset, mean_df=None):
        mean = arr.mean()
        new_nan = []
        if case == 'start':
            if False: # long engough
                first = arr[y + 1 - offset]
                second = arr[y + 2 - offset]
                for i in range(y, x - 1, -1):
                    new_value = first + (first - second) * (y - i + 1) * (1/3)
                    new_nan.append((i, new_value))
            else:
                new_value = arr[y + 1]
                #new_value = mean
                for i in range(x, y + 1):
                    new_nan.append((i, new_value))
        elif case == 'mid':
            first = arr[x - 1 - offset]
            second = arr[y + 1 - offset]
            t_diff = y - x + 2
            for i in range(x, y + 1):
                new_value = first + (second - first) / t_diff * (i - x + 1)
                new_nan.append((i, new_value))
        elif case == 'end':
            new_value = arr[x - 1 - offset]
            for i in range(x, y + 1):
                new_nan.append((i, new_value))
        elif case == 'nothing':
            if mean_df is None or type(mean_df) == int or type(mean_df) == float:
                if mean_df is None:
                    mean_df = 0
                for i in range(x, y + 1):
                    new_nan.append((i, mean_df))
            else:
                # Fill with mean
                mean_df[abs((mean_df - mean_df.mean())/mean_df.std()) > 0.75] = np.nan
                nans, fill = nan_helper(mean_df)
                try:
                    mean_df[nans] = np.interp(fill(nans), fill(~nans), mean_df[~nans])
                except:
                    pass
                if len(mean_df) >= 3:  # Smooth if filled nan with mean
                    mean_df = smooth(mean_df, 3, window='flat')

                for i in range(x, y + 1):
                    new_nan.append((i, mean_df[i]))
        else:
            print('dfghj')
        return new_nan

    def _fill_linear_approx_column(self, arr, partial_df, col, other_cols, x, y, offset, entire_mean_df=None):
        case_nothing = len(partial_df[other_cols + [col]].dropna()) == 0
        new_nan = []
        if case_nothing: # fill with industry approx
            entire_mean_df = entire_mean_df.copy()
            train_X = entire_mean_df[other_cols + [col]].dropna()[other_cols].values
            train_y = entire_mean_df[other_cols + [col]].dropna()[[col]].values
            pred_X = partial_df[other_cols].values
            pred_X = np.append(np.ones((pred_X.shape[0], 1)), pred_X, axis=1)
            train_X = np.append(np.ones((train_X.shape[0], 1)), train_X, axis=1)
            if len(train_y) == 0 and len(train_X) == 0:
                return []
            reg = LinearRegression().fit(train_X, train_y)
            fill_arr = reg.predict(pred_X)

        else: # use existing data to fill
            train = partial_df[other_cols + [col]].dropna()
            train_X = train[other_cols].values
            train_y = train[[col]].values

            pred_idx = partial_df[col].isnull()
            pred_idx = pred_idx[pred_idx == True].index
            pred_idx = partial_df.index.get_indexer(pred_idx)
            pred_X = partial_df.iloc[pred_idx][other_cols].values

            pred_X = np.append(np.ones((pred_X.shape[0], 1)), pred_X, axis=1)
            train_X = np.append(np.ones((train_X.shape[0], 1)), train_X, axis=1)

            reg = LinearRegression().fit(train_X, train_y)
            fill_arr = reg.predict(pred_X)
            #arr[x:y+1] = fill_arr.flatten()


        for i in range(x, y + 1):
            new_nan.append((i, float(fill_arr[i-x])))

        return new_nan






    def _check_quaterly(self, arr):
        quarterly = False
        last_value = None
        value_count = 0
        last_value4 = None
        value4_count = 0
        for i in range(len(arr)):
            value = arr[i]
            if i + 4 > len(arr) - 1:
                break
            value_4 = arr[i + 4]
            if last_value is not None and abs(last_value - value) < 0.001:
                value_count += 1
            else:
                value_count = 0
            last_value = value
            if last_value is not None and last_value4 is not None and abs(last_value4 - value_4) < 0.001 and abs(last_value4 - value) > 0.001:
                value4_count += 1
            else:
                value4_count = 0
            last_value4 = value_4
            if (value4_count == 3 and value_count > 1) or (value4_count > 1 and value_count == 3) or (value_count + value4_count >= 4):
                quarterly = True
                break
        return quarterly



    def _get_nan_blocks(self, arr):
        """
        Get list of dict with partial arrays of nan to fill

        >>> import numpy as np
        >>> self = dataset_nan_fill(df=pd.DataFrame())
        >>> a = np.array([np.nan, np.nan, 2, 3, np.nan, np.nan, np.nan, 4, 5, 6, np.nan])
        >>> self._get_nan_blocks(arr=a)
        [{'i': 0, 'idx': (0, 1), 'arr': array([nan, nan,  2.,  3.]), 'count_filled': 2, 'count_nan': 2, 'case': 'start'}, {'i': 1, 'idx': (4, 6), 'arr': array([ 2.,  3., nan, nan, nan,  4.,  5.,  6.]), 'count_filled': 5, 'count_nan': 3, 'case': 'mid'}, {'i': 2, 'idx': (10, 10), 'arr': array([ 4.,  5.,  6., nan]), 'count_filled': 3, 'count_nan': 1, 'case': 'end'}]
        >>> b = np.array([np.nan, np.nan, np.nan])
        >>> self._get_nan_blocks(arr=b)
        [{'i': 0, 'idx': (0, 2), 'arr': array([nan, nan, nan]), 'count_filled': 0, 'count_nan': 3, 'case': 'nothing'}]
        >>> c = np.array([np.nan])
        >>> self._get_nan_blocks(arr=c)
        [{'i': 0, 'idx': (0, 0), 'arr': array([nan]), 'count_filled': 0, 'count_nan': 1, 'case': 'nothing'}]
        """
        nan_where = np.where(np.isnan(arr))[0]
        nan_blocks = []

        tmp_first_nan = nan_where[0]
        for i in range(1, len(nan_where)):
            if nan_where[i] - nan_where[i - 1] != 1:
                nan_blocks.append((tmp_first_nan, nan_where[i - 1]))
                tmp_first_nan = nan_where[i]
        nan_blocks.append((tmp_first_nan, nan_where[-1]))

        result = []
        prev_y = 0
        for i, (x, y) in enumerate(nan_blocks):
            tmp = {'i': i, 'idx': (x, y)}
            if i + 1 > len(nan_blocks) - 1:
                next_x = len(arr)
            else:
                next_x = nan_blocks[i + 1][0]
            tmp_arr = arr[prev_y:next_x]
            tmp['offset'] = prev_y
            prev_y = y + 1
            tmp['arr'] = tmp_arr
            tmp['count_filled'] = np.count_nonzero(~np.isnan(tmp_arr))
            tmp['count_nan'] = len(tmp_arr) - tmp['count_filled']
            if tmp['count_filled'] == 0:
                tmp['case'] = 'nothing'
            elif x == 0:
                tmp['case'] = 'start'
            elif y == len(arr) - 1:
                tmp['case'] = 'end'
            else:
                tmp['case'] = 'mid'
            result.append(tmp)
        return result



def _data_quality_filter(df, drop_threshold_row_pct=0.0, drop_threshold_row_quantile=0.0, drop_threshold_col_pct=0.0, required_filled_cols=[], append_data_quality_col=True):
    len_df = len(df)

    col_count = df.count(axis=1)
    max_col = max(col_count)
    col_count /= max_col

    row_count = df.count(axis=0)
    max_row = max(row_count)
    row_count /= max_row
    drop_cols = str(df.columns[row_count < drop_threshold_col_pct].tolist())[1:-1]
    dropped_rows_because_required_cols = len(df) - df[required_filled_cols].notna().all(axis=1).value_counts()[True]


    df = df.loc[df[required_filled_cols].notna().all(axis=1)]  # drop row based on required filled columns
    df = df.loc[col_count >= drop_threshold_row_pct]  # Row dropping by percent
    df = df.loc[col_count >= col_count.quantile(drop_threshold_row_quantile)]  # Row dropping by quantile
    df = df.loc[:, df.columns[row_count >= drop_threshold_col_pct]]  # column dropping by percent

    if append_data_quality_col:
        df['data_quality'] = col_count

    info_text = f'Data quality has dropped {dropped_rows_because_required_cols} rows because of required filled cols ({str(required_filled_cols)[1:-1]}) before NaN-filling resulting dataset with {len_df - dropped_rows_because_required_cols} ({int((len_df - dropped_rows_because_required_cols) / len_df * 100)}% of initial dataset size).\n'
    info_text += f'Data quality has dropped {len_df - len(df) - dropped_rows_because_required_cols} rows by dropping every row filled less than {int(max(drop_threshold_row_pct, col_count.quantile(drop_threshold_row_quantile))*100)}% resulting in {len(df)} rows in the end ({int(len(df)/len_df*100)}% of initial dataset size).\n'
    info_text += f'Data quality has dropped {len(row_count) - (row_count >= drop_threshold_col_pct).value_counts()[True]} of {len(row_count)} columns by using threshold of {int(drop_threshold_col_pct * 100)}%. (columns dropped: {drop_cols})\n'


    return df, info_text









def get_clean_data(data_version, recache_raw_data=False, redo_data_cleaning=False, comp_col='ric', time_cols=['data_year', 'data_qrt'], industry_col='industry', required_filled_cols_before_filling=[], required_filled_cols_after_filling=[], drop_threshold_row_pct=0.25, drop_threshold_row_quantile=0.2, drop_threshold_col_pct=0, append_data_quality_col=False):

    cache_folder = os.path.join(my.get_project_directories(key='cache_dir'), 'cleaned_data')
    my_hash = my.data_hash(data_version, comp_col, time_cols, industry_col, required_filled_cols_before_filling, required_filled_cols_after_filling, drop_threshold_row_pct, drop_threshold_row_quantile, drop_threshold_col_pct, append_data_quality_col)
    cache_file = os.path.join(cache_folder, my_hash + '.csv')

    if redo_data_cleaning or not os.path.exists(cache_file):
        print('Cleaned data not cached...')

        df, data_file, data_props, fillnan_formulas = _download_data_from_sql(data_version=data_version, recache=recache_raw_data)

        info_text = f'Initial dataset length {df.shape[0]} rows with {df.shape[1]} columns.\n'
        print(info_text)
        initial_len_df = len(df)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)


        # Count filled columns per row as data-quality metric
        df, tmp_info_text = _data_quality_filter(df, drop_threshold_row_pct=drop_threshold_row_pct, drop_threshold_row_quantile=drop_threshold_row_quantile, drop_threshold_col_pct=drop_threshold_col_pct, required_filled_cols=required_filled_cols_before_filling, append_data_quality_col=append_data_quality_col)
        print(tmp_info_text)

        info_text += tmp_info_text

        dataset_nan_fill(df, company_col=comp_col, time_cols=time_cols, industry_col=industry_col, data_props = data_props, fillnan_formulas=fillnan_formulas, formula_iterations=3)

        len_df = len(df)
        df = df.loc[df[required_filled_cols_after_filling].notna().all(axis=1)]  # drop row based on required filled columns

        tmp_info_text = f'Data quality has dropped {len_df - len(df)} rows because of required filled cols ({str(required_filled_cols_after_filling)[1:-1]}) after NaN-filling resulting dataset with {len(df)} ({int(len(df) / initial_len_df * 100)}% of initial dataset size).'
        print(tmp_info_text)
        info_text += tmp_info_text

        with open(cache_file[:-3] + 'info', "w") as text_file:
            text_file.write(info_text)
        df.to_csv(cache_file)

    else:
        print('Cleaned data already cached.')
        with open(cache_file[:-3] + 'info', "r") as text_file:
            info_text = text_file.read()
        df = pd.read_csv(cache_file)
        print(info_text)

    return df



if  __name__ == '__main__':
    # Working directory must be the higher .../app folder
    from app.z_helpers import helpers as my
    my.convenience_settings()

    dataset_name = 'handpicked_dataset'

    from app.b_data_cleaning import get_dataset_registry
    dataset_props = get_dataset_registry()[dataset_name]


    recache_raw_data = False
    redo_cleaning = False
    comp_col = dataset_props['company_col']
    time_cols = dataset_props['iter_cols']
    industry_col = dataset_props['industry_col']

    drop_row_if_col_not_filled_before_filling = ['sales', 'eps']
    drop_row_if_col_not_filled_after_filling = ['ebit']

    df = get_clean_data(data_version=dataset_name, recache_raw_data=recache_raw_data, redo_data_cleaning=redo_cleaning, comp_col=comp_col, time_cols=time_cols, industry_col=industry_col, required_filled_cols_before_filling=drop_row_if_col_not_filled_before_filling, required_filled_cols_after_filling=drop_row_if_col_not_filled_after_filling)

    print(df)
