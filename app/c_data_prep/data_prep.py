import os, sys, zlib, hashlib, warnings, math, datetime
import psycopg2, psycopg2.extras, pickle, progressbar, time, hashlib
import pandas as pd
import numpy as np
import tensorflow as tf

### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import z_helpers as my


##################################


def _download_data_from_sql(data_version='final_data', recache=False):
    data_dict = {'final_data': 'final_data',
                 'final_data_2': 'final_data_2'}
    query = "SELECT * FROM {}".format(data_dict[data_version])

    param_dic = my.get_credentials(credential='aws')

    cache_folder = os.path.join(my.get_project_directories(key='cache_dir'), 'raw_data')
    data_file = os.path.join(cache_folder, (data_version + '.csv'))
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    if recache or not os.path.exists(data_file):
        print('Getting raw data via sql...')

        with my.postgresql_connect(param_dic) as conn:
            df = pd.read_sql_query(query, con=conn)
            df.to_csv(data_file, index=False)
        print('Raw data cached.')

    else:
        print('Raw data already cached.')
        df = pd.read_csv(data_file, index_col=False)

    return df, data_file





class data_prep:
    def __init__(self, dataset='final_data_2', recache=False, keep_raw_cols='default', drop_cols='default'):
        self.dataset = dataset
        self.recache = recache

        if keep_raw_cols == 'default':
            keep_raw_cols = []
        self.keep_raw_cols = keep_raw_cols

        if drop_cols == 'default':
            drop_cols = ['ric', 'curcdq', 'rp', 'updq', 'iid', 'exchg', 'costat', 'fic', 'srcq', 'curncdq', 'acctsdq', 'acctstdq', 'ggroup']

        self.cols_drop = drop_cols
        self.cols_just_these = False

        self.comps_exclude = []
        self.comps_just_these = False

        self.lagged_col_dict = {}

        self.computed = False


    def _get_raw_data(self):
        df, self.raw_file_path = _download_data_from_sql(data_version=self.dataset, recache=self.recache)

        dataset_pops = {'final_data':   {'iter_col': ['period_year', 'period_qrt'], 'company_col': 'gvkey', 'y_col': ['tr_f_ebit'], 'category_cols': ['gsector', 'ggroup'], 'date_cols': []},
                        'final_data_2': {'iter_col': ['period_year', 'period_qrt'], 'company_col': 'gvkey', 'y_col': ['tr_f_ebit'], 'category_cols': ['gsector', 'ggroup'], 'date_cols': []}}

        self.dataset_iter_col = dataset_pops[self.dataset]['iter_col']
        self.dataset_company_col = dataset_pops[self.dataset]['company_col']
        self.dataset_y_col = dataset_pops[self.dataset]['y_col']
        self.dataset_category_cols = dataset_pops[self.dataset]['category_cols']
        self.dataset_date_cols = dataset_pops[self.dataset]['date_cols']



        # Format the data to the correct format
        for cat in self.dataset_category_cols:
            df[cat] = df[cat].astype('category')
        for dt_col in self.dataset_date_cols:
            df[dt_col] = pd.to_datetime(df[dt_col])

        df[self.dataset_iter_col] = df[self.dataset_iter_col].astype(int)

        i_col = pd.DataFrame()
        for col in self.dataset_iter_col:
            i_col[col] = df[col].astype(int).astype(str).str.zfill(2)
        df['iter_col'] = i_col.agg(''.join, axis=1).astype(int)
        self.iter_idx = np.array(df['iter_col'].unique()).tolist()
        df.set_index(keys='iter_col', inplace=True)
        df.sort_index(inplace=True)

        self.raw_data = df.copy()
        self.mod_data = df.copy()



    def _transform_dummy_vars(self):
        cat_cols = self.raw_data.select_dtypes(include=['object', 'category']).columns.tolist()
        df = self.raw_data.copy()
        dummy_dict = {}
        for col in cat_cols:
            tmp = pd.get_dummies(df[col], dummy_na=True, drop_first=True, prefix=str(col))
            dummy_dict[col] = tmp.columns.tolist()
            df = pd.concat([df, tmp], axis=1)
        self.data = df.drop(columns=cat_cols)
        self.dummy_col_dict = dummy_dict



    #############################################################

    def _block_split(self, df):

        # Get time seperators
        t = self.iter_idx
        len_t = len(t)
        len_window = self.window_pred_width + self.window_input_width

        step_size = 1
        t_lower = [t[i] for i in list(range(0, len_t - len_window + 1, step_size))]
        t_higher = [t[i] for i in list(range(len_window - 1, len_t, step_size))]
        val_split = len(t_lower) - int(len(t_lower) * (self.split_val_time_size + self.split_test_time_size))
        test_split = len(t_lower) - int(len(t_lower) * (self.split_test_time_size))
        t_boundaries = {'train': {'upper':t_higher[val_split]}, 'val':{'lower':t_lower[val_split], 'upper':t_higher[test_split]}, 'test':{'lower': t_lower[test_split]}}

        # Get comp seperators
        comp_list = df[self.dataset_company_col].unique().tolist()
        val_i, test_i = int((1 - self.split_val_comp_size - self.split_test_comp_size) * len(comp_list)), int((1 - self.split_test_comp_size) * len(comp_list))
        val_comp_list, test_comp_list = comp_list[val_i:test_i], comp_list[test_i:]

        val_df = df[(df[self.dataset_company_col].isin(val_comp_list) | ((df.index < t_boundaries['val']['upper']) & (df.index >= t_boundaries['val']['lower'])))]
        test_df = df[(df[self.dataset_company_col].isin(test_comp_list) | (df.index >= t_boundaries['test']['lower']))]
        train_df = df[(~df[self.dataset_company_col].isin(val_comp_list) & ~df[self.dataset_company_col].isin(test_comp_list) & (df.index < t_boundaries['train']['upper']))]

        print(
            'Total dataset size: {}\n{} samples for training ({}%) / {} samples for validation ({}%) / {} samples for testing ({}%)'.format(
                len(self.raw_data), len(train_df), int(len(train_df) / len(self.raw_data) * 100),
                len(val_df), int(len(val_df) / len(self.raw_data) * 100), len(test_df),
                int(len(test_df) / len(self.raw_data) * 100)))

        return train_df, val_df, test_df



    def _get_idx_dict_block_rolling_window(self, idx_col, hist_periods_in_block=4, val_time_steps=1, test_time_steps=1, subpress_warning=False):
        hist_periods_in_block = self.window_input_width + hist_periods_in_block - 1
        if hist_periods_in_block == self.window_input_width and subpress_warning is False:
            warnings.warn('You are effectivly using single_time_rolling because the hist_periods_in_block is equal to the window_input_width!')
        elif hist_periods_in_block < self.window_input_width:
            raise Exception(f'hist_periods_in_block has {hist_periods_in_block} periods must be >= window_input_width with {self.window_input_width} periods')

        idx_list = list(idx_col.unique())
        idx_len = len(idx_list)
        input_width = self.window_input_width
        pred_width = self.window_pred_width
        shift = self.window_shift
        window_len = hist_periods_in_block + (pred_width * 3) + (val_time_steps + test_time_steps - 2)
        sample_len = input_width + pred_width

        idx_lower = list(range(0, idx_len - window_len + 1, shift))
        idx_upper = list(range(window_len - 1, idx_len, shift))

        idx_dict = {}
        for lower, upper in zip(idx_lower, idx_upper):
            train_list = self._all_rolling_windows_in_block(idx_list=idx_list, start=lower, end=(upper - pred_width - pred_width - val_time_steps - test_time_steps + 1) + 1)
            val_list = self._all_rolling_windows_in_block(idx_list=idx_list, start=(upper - sample_len - test_time_steps - pred_width - val_time_steps + 3), end=(upper - pred_width - test_time_steps + 1))
            test_list = self._all_rolling_windows_in_block(idx_list=idx_list, start=(upper - sample_len - test_time_steps + 2), end=upper)
            all_list = [(i[0], j[1]) for i, j in zip(train_list, test_list)]

            tmp_dict = {'__all__': all_list,
                        'train': train_list,
                        'val': val_list,
                        'test': test_list}
            idx_dict[f'{idx_list[lower]}_{idx_list[upper]}'] = tmp_dict

        return idx_dict



    def _all_rolling_windows_in_block(self, idx_list, start, end, raw=False):
        idx_len = len(idx_list)
        input_width = self.window_input_width
        pred_width = self.window_pred_width
        shift = self.window_shift
        window_len = input_width + pred_width

        idx_lower = list(range(start, end - window_len + 2, shift))
        idx_upper = list(range(start + window_len - 1, end + 1, shift))

        if raw:
            window_list = [(i, j) for i, j in zip(idx_lower, idx_upper)]
        else:
            window_list = [(idx_list[i], idx_list[j]) for i, j in zip(idx_lower, idx_upper)]

        return window_list





    def _normalization_multicore_function(self, df, idx_lower, idx_upper, i=0):
        relevant_cols = [i for i in df.columns.tolist() if (i not in [item for sublist in self.dummy_col_dict.values() for item in sublist])]
        norm_cols = [i for i in relevant_cols if (i not in self.dataset_iter_col) and (i != self.dataset_company_col)]
        df = df[relevant_cols]

        comp_list = df[self.dataset_company_col].unique().tolist()
        norm_param = {}

        tmp_df = df[(df.index >= idx_lower) & (df.index <= idx_upper)]
        tmp_mean = tmp_df[norm_cols].mean()
        tmp_std = tmp_df[norm_cols].std()
        tmp_mean_np = my.custom_hdf5.pd_series_to_2d_array(pd_series=tmp_mean)
        tmp_std_np = my.custom_hdf5.pd_series_to_2d_array(pd_series=tmp_std)
        norm_param['__all__'] = {'mean': tmp_mean_np, 'std': tmp_std_np}

        for comp in comp_list:
            tmp_df = df[(df[self.dataset_company_col] == comp) & (df.index >= idx_lower) & (df.index <= idx_upper)]
            tmp_mean = tmp_df[norm_cols].mean()
            tmp_std = tmp_df[norm_cols].std()
            tmp_mean_np = my.custom_hdf5.pd_series_to_2d_array(pd_series=tmp_mean)
            tmp_std_np = my.custom_hdf5.pd_series_to_2d_array(pd_series=tmp_std)
            norm_param[f'c_{comp}'] = {'mean': tmp_mean_np, 'std': tmp_std_np}

        print(f' {i}', end='')

        return norm_param

    def _get_normalization_param(self, df, idx_dict):
        norm_get = {}
        for _, time_dict in idx_dict.items():
            for i, j in time_dict['train']:
                norm_get[f't_{i}_{j}'] = (i, j)


        print('\nCaching indices/iterators and normalization parameters:')

        import multiprocessing

        max_i = len(norm_get)
        num_workers = multiprocessing.cpu_count()

        input_data_pairs = []
        i = 1
        for key, idx in norm_get.items():
            input_data_pairs.append((df, idx[0], idx[1], i))
            i += 1

        start = datetime.datetime.now()

        print(f'Multiprocessing with {num_workers} simultaneous processes. Start time {start.strftime("%H:%M:%S")}')
        print(f'Out of {max_i} tasks these are done:', end='')



        # Multi-Core Multiprocessing
        p = multiprocessing.Pool(processes=num_workers)
        data = p.starmap(self._normalization_multicore_function, input_data_pairs)
        p.close()

        norm_param_dict = dict(zip(list(norm_get.keys()), data))
        end = datetime.datetime.now()

        print(f'\nIndices/iterators and normalization parameters cached. Needed {int((end-start).seconds/60)}:{int((end-start).seconds%60)} mins with end time {end.strftime("%H:%M:%S")}')

        return norm_param_dict




    def _get_data_hash(self, *args):
        str_args = (str(args)[1:-1]).replace("'", "").replace(", ", "/")
        hash = hashlib.shake_256(str_args.encode()).hexdigest(5)
        return hash



    def compute(self):
        # Raise warnings for stupid data combinations
        if self.split_method == 'single_time_rolling' and 'ToDo: REPLACE TEST' != '__all__':
            warnings.warn('Having a rolling window and single_company results in ONE TRAINING SET per time which is stupid!')


        # Transform data
        self._get_raw_data()
        self.companies = self.raw_data[self.dataset_company_col].unique().tolist()
        self.comps = self.companies
        self.cols = self.raw_data.columns.tolist()

        # Transform categorical variables to dummy
        self._transform_dummy_vars()

        cache_folder = my.get_project_directories(key='cache_dir')
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)


        if self.split_method == 'block_rolling':
            hist_periods_in_block = self.split_props['train_time_steps']
            val_time_steps = self.split_props['val_time_steps']
            test_time_steps = self.split_props['test_time_steps']

        elif self.split_method == 'single_time_rolling':
            hist_periods_in_block = 1
            val_time_steps = self.split_props['val_time_steps']
            test_time_steps = self.split_props['test_time_steps']

        elif self.split_method == 'block_static':
            train_df, val_df, test_df = self._block_split(df=self.data)

        else:
            raise Exception(f'UNKNOWN Split method {self.split_method}.')



        cache_hash = self._get_data_hash(self.dataset, self.data.index.unique().tolist(), self.window_input_width, self.window_pred_width, self.window_shift, hist_periods_in_block, val_time_steps, test_time_steps, self.lagged_col_dict)
        norm_cache_file = os.path.join(cache_folder, f'norm_parm_{cache_hash}.hdf5')
        iter_cache_file = os.path.join(cache_folder, f'iter_dict_{cache_hash}.pkl')

        if self.recache or not os.path.exists(norm_cache_file):

            idx_dict = self._get_idx_dict_block_rolling_window(idx_col=self.data.index, hist_periods_in_block=hist_periods_in_block, val_time_steps=val_time_steps, test_time_steps=test_time_steps)
            norm_dict = self._get_normalization_param(df=self.data, idx_dict=idx_dict)

            print('\nDumping cache to file for next use...')
            pickle.dump(idx_dict, open(iter_cache_file, 'wb'))
            my.custom_hdf5.dict_to_hdf5(file=norm_cache_file, save_dict=norm_dict)
            del norm_dict

            print('Iterators/indices and normalization parameters cached.')

        else:

            idx_dict = pickle.load(open(iter_cache_file, 'rb'))
            print('Iterators/indices and normalization parameters already cached.')


        self.iter_dict = idx_dict
        self.norm_param_file = norm_cache_file

        self.data_hash = cache_hash

        self.computed = True

        print('Data class skeleton constructed (computed)! Ready to iterate across or subscript...')



    def _prep_final_dataset(self, df, norm_mean, norm_std):
        df = df.copy()

        sort_cols = self.dataset_iter_col + [self.dataset_company_col]
        df.sort_values(sort_cols, inplace=True)

        warning = None
        if len(df) > (self.window_input_width + self.window_pred_width):
            warning = [str(df[self.dataset_company_col].unique().tolist())[1:-1], df.index.min(), df.index.max(), (f'Company {str(df[self.dataset_company_col].unique().tolist())[1:-1]} has data duplicates in time_step {df.index.min()}-{df.index.max()}')]
            df = df.drop_duplicates(subset=sort_cols, keep='last')

        cols = self.cols

        if self.cols_just_these == False:
            final_cols = [i for i in cols if (i not in self.cols_drop) and (i not in self.dataset_iter_col) and (i != self.dataset_company_col) and (i not in self.dataset_date_cols)]
        else:
            if type(self.cols_just_these) != list:
                raise Exception('cols_just_these has to be a list of column names')
            final_cols = [i for i in self.cols_just_these if (i not in self.dataset_iter_col) and (i != self.dataset_company_col) and (i not in self.dataset_date_cols)]

        norm_cols = [i for i in df.columns.tolist() if (i in final_cols) and (i not in self.norm_keep_raw)]
        df[norm_cols] = (df[norm_cols] - norm_mean[norm_cols].fillna(0)) / norm_std[norm_cols].fillna(1)

        for key, value in self.dummy_col_dict.items():
            try:
                final_cols.remove(key)
                final_cols.extend(value)
            except:
                pass

        X = pd.DataFrame(df.iloc[:-self.window_pred_width][final_cols])
        y = pd.DataFrame(df.iloc[-self.window_pred_width:][self.dataset_y_col])

        return X, y, warning



    def _final_dataset(self, train_dict, val_dict, test_dict, iter_step, data_hash):

        final_data_cache_folder = os.path.join(my.get_project_directories(key='cache_dir'), data_hash)
        if not os.path.exists(final_data_cache_folder):
            os.makedirs(final_data_cache_folder)

        column_hash = self._get_data_hash(self.cols_drop, self.dataset_date_cols, self.cols_just_these, self.norm_keep_raw, self.comps_just_these, self.normalize_method, self.comps_exclude)
        final_file = os.path.join(final_data_cache_folder, f'{column_hash}_{iter_step}_iter-step.npz')

        if os.path.exists(final_file):
            print('Iteration-step already cached just getting data from file...')

            loaded = np.load(final_file)

            train_X = loaded['train_X']
            train_y = loaded['train_y']
            train_idx = loaded['train_idx']
            val_X = loaded['val_X']
            val_y = loaded['val_y']
            val_idx = loaded['val_idx']
            test_X = loaded['test_X']
            test_y = loaded['test_y']
            test_idx = loaded['test_idx']
            ndarray_columns = pickle.load(open((final_file[:-4] + '_cols.pkl'), 'rb'))


            print('Got data from cached file.')

        else:
            if self.normalize_method == 'block':
                mean = 0
                std = 1

            # Progressbar to see hwo long it will still take
            print(f'\nCaching, normalizing, and preparing data for iteration-step {iter_step}:')
            time.sleep(0.5)
            widgets = ['[',
                       progressbar.Timer(format='elapsed: %(elapsed)s'),
                       '] ',
                       progressbar.Bar('█'), ' (',
                       progressbar.ETA(), ') ',
                       ]
            progress_bar = progressbar.ProgressBar(max_value=((len(self.companies) - len(self.comps_exclude)) * (len(train_dict) + len(val_dict) + len(test_dict)) if self.comps_just_these == False else (len(train_dict) + len(val_dict) + len(test_dict)) * len(self.comps_just_these)), widgets=widgets).start()

            warning_list = []

            ##################### TRAIN #####################

            train_X = []
            train_y = []
            train_idx = []
            i = 0
            for lower, upper in train_dict:
                norm_key = f't_{lower}_{upper}'

                if self.normalize_method == 'time-step':
                    mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, '__all__', 'mean')
                    std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, '__all__', 'std')

                comp_iter_list = my.custom_hdf5.get_comp_list(file=self.norm_param_file, norm_key=norm_key)
                if self.comps_just_these == False:
                    comp_iter_list = [i for i in comp_iter_list if (i not in self.comps_exclude) and (i != '__all__')]
                else:
                    if type(self.comps_just_these) != list:
                        raise Exception('comp_iter_list has to be a list of column names')
                    comp_iter_list = self.comps_just_these


                X_tmp_col_compare = False
                y_tmp_col_compare = False
                for comp in comp_iter_list:
                    comp = type(self.data[self.dataset_company_col].iloc[0])(comp[2:])
                    if self.normalize_method == 'set':
                        mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, f'c_{comp}', 'mean')
                        std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, f'c_{comp}', 'std')

                    df_set = self.data[(self.data.index >= lower) & (self.data.index <= upper) & (self.data[self.dataset_company_col] == comp)]
                    if len(df_set) > 0:
                        X, y, warning = self._prep_final_dataset(df=df_set, norm_mean=mean, norm_std=std)
                        warning_list.append(warning) if warning is not None else None
                        if X_tmp_col_compare != False:
                            if X_tmp_col_compare != X.columns.tolist():
                                raise Exception(f'X columns are not identical across company iteration.\nGot: [{X.columns.tolist()}]\nExpected: [{X_tmp_col_compare}]')
                            if y_tmp_col_compare != y.columns.tolist():
                                raise Exception(f'Y columns are not identical across company iteration.\nGot: [{y.columns.tolist()}]\nExpected: [{y_tmp_col_compare}]')
                        else:
                            X_tmp_col_compare = X.columns.tolist()
                            y_tmp_col_compare = y.columns.tolist()


                        if len(X) == self.window_input_width and len(y) == self.window_pred_width:
                            train_X.append(X.values)
                            train_y.append(y.values)
                            train_idx.append([comp, lower, upper])
                    i += 1
                    progress_bar.update(i)

            train_X = np.asarray(train_X)
            train_y = np.asarray(train_y)
            train_idx = np.asarray(train_idx)



            ##################### VAL #####################

            val_X = []
            val_y = []
            val_idx = []
            for lower, upper in val_dict:
                norm_key = f't_{lower}_{upper}'

                if self.normalize_method == 'time-step':
                    mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, '__all__', 'mean')
                    std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, '__all__', 'std')


                for comp in comp_iter_list:
                    comp = type(self.data[self.dataset_company_col].iloc[0])(comp[2:])
                    if self.normalize_method == 'set':
                        mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, f'c_{comp}', 'mean')
                        std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, f'c_{comp}', 'std')

                    df_set = self.data[(self.data.index >= lower) & (self.data.index <= upper) & (self.data[self.dataset_company_col] == comp)]
                    if len(df_set) > 0:
                        X, y, warning = self._prep_final_dataset(df=df_set, norm_mean=mean, norm_std=std)
                        warning_list.append(warning) if warning is not None else None

                        if X_tmp_col_compare != X.columns.tolist():
                            raise Exception(f'X columns are not identical across company iteration.\nGot: [{X.columns.tolist()}]\nExpected: [{X_tmp_col_compare}]')
                        if y_tmp_col_compare != y.columns.tolist():
                            raise Exception(f'Y columns are not identical across company iteration.\nGot: [{y.columns.tolist()}]\nExpected: [{y_tmp_col_compare}]')


                        if len(X) == self.window_input_width and len(y) == self.window_pred_width:
                            val_X.append(X.values)
                            val_y.append(y.values)
                            val_idx.append([comp, lower, upper])
                    i += 1
                    progress_bar.update(i)

            val_X = np.asarray(val_X)
            val_y = np.asarray(val_y)
            val_idx = np.asarray(val_idx)


            ##################### TEST #####################

            test_X = []
            test_y = []
            test_idx = []
            for lower, upper in test_dict:
                norm_key = f't_{lower}_{upper}'

                if self.normalize_method == 'time-step':
                    mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, '__all__', 'mean')
                    std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, '__all__', 'std')

                for comp in comp_iter_list:
                    comp = type(self.data[self.dataset_company_col].iloc[0])(comp[2:])
                    if self.normalize_method == 'set':
                        mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, f'c_{comp}', 'mean')
                        std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, f'c_{comp}', 'std')

                    df_set = self.data[(self.data.index >= lower) & (self.data.index <= upper) & (self.data[self.dataset_company_col] == comp)]
                    if len(df_set) > 0:
                        X, y, warning = self._prep_final_dataset(df=df_set, norm_mean=mean, norm_std=std)
                        warning_list.append(warning) if warning is not None else None

                        if X_tmp_col_compare != X.columns.tolist():
                            raise Exception(f'X columns are not identical across company iteration.\nGot: [{X.columns.tolist()}]\nExpected: [{X_tmp_col_compare}]')
                        if y_tmp_col_compare != y.columns.tolist():
                            raise Exception(f'Y columns are not identical across company iteration.\nGot: [{y.columns.tolist()}]\nExpected: [{y_tmp_col_compare}]')

                        if len(X) == self.window_input_width and len(y) == self.window_pred_width:
                            test_X.append(X.values)
                            test_y.append(y.values)
                            test_idx.append([comp, lower, upper])
                    i += 1
                    progress_bar.update(i)

            test_X = np.asarray(test_X)
            test_y = np.asarray(test_y)
            test_idx = np.asarray(test_idx)

            ndarray_columns = {'X': dict(zip(range(len(X_tmp_col_compare)), X_tmp_col_compare)),
                               'y': dict(zip(range(len(y_tmp_col_compare)), y_tmp_col_compare))}
            pickle.dump(ndarray_columns, open((final_file[:-4] + '_cols.pkl'), 'wb'))
            warning_list = pd.DataFrame(warning_list, columns=["Company", "Lower IDX", "Upper IDX", "Message"])
            warning_list.drop_duplicates(keep='first', inplace=True, ignore_index=True)
            warning_list.to_csv((final_file[:-4] + '_warnings.csv'), index=False)

            np.savez_compressed(final_file, train_X=train_X, train_y=train_y, train_idx=train_idx, val_X=val_X, val_y=val_y, val_idx=val_idx, test_X=test_X, test_y=test_y, test_idx=test_idx)

            print(f'\nData cached for iteration-step {iter_step}.')


        OUT = {'train': {'X': train_X, 'y': train_y, 'idx': train_idx},
               'val':   {'X': val_X,   'y': val_y,   'idx': val_idx},
               'test':  {'X': test_X,  'y': test_y,  'idx': test_idx},
               'columns': ndarray_columns}
        self.latest_out = OUT
        return OUT


    ############# ITERABLE #############

    def __iter__(self):
        if self.computed is False:
            self.compute()
        self._custom_iter_ = iter(self.iter_idx)
        return self

    def __next__(self):
        current = next(self._custom_iter_)
        train_np, val_np, test_np = self._final_dataset(train_dict=current['train'], val_dict=current['val'], test_dict=current['test'])
        #if self.current < self.high:
        #    return self.current
        #raise StopIteration
        return train_np, val_np, test_np

    ####################################


    ########### SUBSCRIPTABLE ###########

    def __getitem__(self, obj):
        if type(obj) == int:
            obj = list(self.iter_idx)[obj]
        data_dict = self._final_dataset(train_dict=self.iter_dict[obj]['train'], val_dict=self.iter_dict[obj]['val'], test_dict=self.iter_dict[obj]['test'], iter_step=obj, data_hash=self.data_hash)
        return data_dict

    def __len__(self):
        return len(self.iter_dict)

    #####################################

    def __str__(self):
        return f"Custom-Data class:\n" \
               f"Dataset: {self.dataset}\n" \
               f"Recache: {self.recache}\n" \
               f"Companies ({len(self.companies)}): {str(self.companies)}\n" \
               f"Time-steps ({len(self.iter_dict)}): {str(list(self.iter_dict.keys()))}\n" \
               f"Normaliation method: {self.normalize_method}\n" \
               f"Split method: {self.split_method}\n" \
               f"Split props: {self.split_props}\n" \
               f"Window props: ({self.window_input_width}, {self.window_pred_width}, {self.window_shift})\n" \
               f"Data hash: {self.data_hash}"

    def details(self):
        print('\n' + self.__str__() + '\n')


    def help(self):
        a = "\n\nTo use this data prep class please use the following steps:\n" \
            "-------------------------------------------------------------\n" \
            "## Define what dataset to use:\n" \
            "data = data_prep(dataset='final_data_2', recache=False, keep_raw_cols='default', drop_cols='default')\n" \
            "\n" \
            "## Define the data window with historical input length/period number, prediction length and step shift:\n" \
            "data.window(input_width=5*4, pred_width=4, shift=1)\n" \
            "\n" \
            "## Define the split between train/val/test. There are three methods avalable:\n" \
            "# - block_static_split(): data divided by percent into three static blocks -> just one dataset\n" \
            "# - block_rolling_split(): data divided by fixed number of periods rolling acoss entire dataset -> iterable & subscriptable\n" \
            "# - single_time_rolling(): just one train/val/test period/window in dataset -> iterable & subscriptable\n" \
            "data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=5*4*2, val_time_steps=1, test_time_steps=1, shuffle=True)\n" \
            "\n" \
            "## Normalize the date (three normalization levels avalable: [entire dataset] block / [across all comps per] time-step / [per company-time] set)\n" \
            "data.normalize(method='time')\n" \
            "\n" \
            "## Get data by either: (if block_static_split just method c avalable)\n" \
            "# a) for i in data:\n" \
            "# b) data['199503_201301']\n" \
            "# c) data.get_data()\n" \
            "\n" \
            "## Finally you can apply column and company filters:\n" \
            "data.column_filter(include_features_list=None, exclude_features_list=None)\n" \
            "data.company_filter(include_comp_list=None, exclude_comp_list=None)\n" \
            "\n\n"
        print(a)



    def export_to_excel(self, path=None):
        print('Exporting to Excel. May take a while because data is huge...')
        if path is None:
            path = my.get_project_directories(key='data_dir')
            path = os.path.join(path, 'final_input_data')
            if not os.path.exists(path):
                os.makedirs(path)
        today = datetime.datetime.now()
        now_str = today.strftime("%Y%m%d_%H-%M")
        file_endings = self.data_hash + '_' + now_str + '.csv'

        for ty in ['train', 'val', 'test']:
            tmp_X = self.latest_out[ty]['X']
            tmp_idx = self.latest_out[ty]['idx']
            tmp_cols = self.latest_out['columns']['X']

            my_array = tmp_X
            my_array = my_array.reshape((-1, my_array.shape[-1]))
            count_idx = np.ceil(np.array(range(1, my_array.shape[0] + 1)) / 20).astype(int).reshape(-1, 1)
            props_dict = pd.DataFrame.from_dict(
                dict(zip(range(1, len(tmp_idx) + 1), tmp_idx.tolist())), orient='index',
                columns=[data.dataset_company_col, 'lower_idx', 'upper_idx'])
            new_array = np.append(count_idx, my_array, 1)
            new_df = pd.DataFrame(new_array[:, 1:], columns=list(tmp_cols.values()), index=new_array[:, 0])
            final_df = pd.concat([new_df, props_dict], axis=1)

            final_df.to_csv(os.path.join(path, (ty + '_' + file_endings)), index=False)
        print(f'Done exporting. Data is here: {path}')




    #############################################################


    def lagged_variables(self, lagged_col_dict={'tr_f_ebit': [-1, -2, -3, -4]}):
        self.computed = False
        self.lagged_col_dict = lagged_col_dict


    #############################################################


    def window(self, input_width=5*4, pred_width=4, shift=1):
        self.computed = False
        self.window_input_width = input_width
        self.window_pred_width = pred_width
        self.window_shift = shift


    #############################################################


    def block_static_split(self, val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True):
        self.computed = False
        self.split_method = 'block_static'
        self.split_props = {'val_comp_size': val_comp_size, 'test_comp_size': test_comp_size,
                            'val_time_size': val_time_size, 'test_time_size': test_time_size, 'shuffle': shuffle}

    def block_rolling_split(self, val_comp_size=0, test_comp_size=0, train_time_steps=4, val_time_steps=1,
                            test_time_steps=1, shuffle=True):
        self.computed = False
        self.split_method = 'block_rolling'
        self.split_props = {'val_comp_size': val_comp_size, 'test_comp_size': test_comp_size,
                            'train_time_steps': train_time_steps, 'val_time_steps': val_time_steps,
                            'test_time_steps': test_time_steps, 'shuffle': shuffle}

    def single_time_rolling(self, val_time_steps=1, test_time_steps=1, shuffle=True):
        self.computed = False
        self.split_method = 'single_time_rolling'
        self.split_props = {'train_time_steps': 1, 'val_time_steps': val_time_steps,
                            'test_time_steps': test_time_steps, 'shuffle': shuffle}


    #############################################################

    def normalize(self, method='year', keep_raw_feature=[]):
        # per 'block': all data in train
        # per 'time'/'time-step': per time-step across all companies
        # per 'set'/'company-time-set': per single data-set -> per company in every time-step
        self.computed = False
        if not hasattr(self, 'split_method'):
            raise Exception('Please first apply test_train_split before normalize!')

        self.norm_keep_raw = keep_raw_feature

        if method == 'block':
            if self.split_method == 'single_time_rolling':
                raise Exception('Combination of single_time_rolling split and block normalization not possible!')
            self.normalize_method = 'block'
        elif method == 'time' or method == 'time-step':
            self.normalize_method = 'time-step'
        elif method == 'set' or method == 'company-time-set':
            self.normalize_method = 'set'
        else:
            raise Exception(f'UNKNOWN method={method} for normalization. Possible are: block / time / set.')


    #############################################################
    ### After compute ###

    def feature_filter(self, include_features_list=None, exclude_features_list=None):
        pass

    def column_filter(self, include_features_list=None, exclude_features_list=None):
        self.feature_filter(include_features_list=include_features_list, exclude_features_list=exclude_features_list)

    def company_filter(self, include_comp_list=None, exclude_comp_list=None):
        if self.normalize_method == 'block':
            warnings.warn('You set a company filter but normalize across the entire block including all companies. The company filter is not applied to the normalization.')
            self.computed = False
        pass

    #############################################################


















if __name__ == '__main__':

    data = data_prep(dataset='final_data_2', recache=False, keep_raw_cols='default', drop_cols='default')

    data.window(input_width=5*4, pred_width=4, shift=1)

    #data.block_static_split(val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=5*4*2, val_time_steps=1, test_time_steps=1, shuffle=True)
    #data.single_time_rolling(val_time_steps=1, test_time_steps=1, shuffle=True)

    # data.normalize(method='block')
    data.normalize(method='time')
    # data.normalize(method='set')

    data.compute()

    data.help()

    #out = data['199503_201301']
    #data.export_to_excel()

    print(data)



    #data.feature_filter(include_features_list=None, exclude_features_list=None)
    #data.company_filter(include_comp_list=None, exclude_comp_list=None)

    #print(train_ds, len(train_ds))





