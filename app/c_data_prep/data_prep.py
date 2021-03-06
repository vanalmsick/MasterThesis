import os, sys, zlib, hashlib, warnings, math, datetime, random
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
        # ToDo: Implement real NaN handling
        self.data = self.data.fillna(self.data.mean().fillna(0))
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





    def _normalization_multicore_function(self, df, idx_lower, idx_upper):
        relevant_cols = [i for i in df.columns.tolist() if (i not in [item for sublist in self.dummy_col_dict.values() for item in sublist])]
        norm_cols = [i for i in relevant_cols if (i not in self.dataset_iter_col) and (i != self.dataset_company_col)]
        df = df[relevant_cols]

        comp_list = df[self.dataset_company_col].unique().tolist()
        norm_param = {}

        tmp_df = df[(df.index >= idx_lower) & (df.index <= idx_upper)]
        tmp_mean = tmp_df[norm_cols].mean().fillna(0)
        tmp_std = tmp_df[norm_cols].std().fillna(1)
        tmp_mean_np = my.custom_hdf5.pd_series_to_2d_array(pd_series=tmp_mean)
        tmp_std_np = my.custom_hdf5.pd_series_to_2d_array(pd_series=tmp_std)
        norm_param['__all__'] = {'mean': tmp_mean_np, 'std': tmp_std_np}

        for comp in comp_list:
            tmp_df = df[(df[self.dataset_company_col] == comp) & (df.index >= idx_lower) & (df.index <= idx_upper)]
            tmp_mean = tmp_df[norm_cols].mean().fillna(0)
            tmp_std = tmp_df[norm_cols].std().fillna(1)
            tmp_mean_np = my.custom_hdf5.pd_series_to_2d_array(pd_series=tmp_mean)
            tmp_std_np = my.custom_hdf5.pd_series_to_2d_array(pd_series=tmp_std)
            norm_param[f'c_{comp}'] = {'mean': tmp_mean_np, 'std': tmp_std_np}

        return norm_param

    def _get_normalization_param(self, df, idx_dict):
        # ToDo: Check if Normalization is OOS
        norm_get = {}
        for _, time_dict in idx_dict.items():
            for i, j in time_dict['train']:
                norm_get[f't_{i}_{j}'] = (i, j)
            for i, j in time_dict['val']:
                norm_get[f't_{i}_{j}'] = (i, j)
            for i, j in time_dict['test']:
                norm_get[f't_{i}_{j}'] = (i, j)


        print('\nCaching indices/iterators and normalization parameters:')

        input_data_pairs = []
        i = 1
        for key, idx in norm_get.items():
            input_data_pairs.append((df, idx[0], idx[1]))
            i += 1

        data = my.multiprocessing_func_with_progressbar(func=self._normalization_multicore_function, argument_list=input_data_pairs, num_processes=-1)
        norm_param_dict = dict(zip(list(norm_get.keys()), data))

        return norm_param_dict




    def _get_data_hash(self, *args):
        str_args = (str(args)[1:-1]).replace("'", "").replace(", ", "/")
        hash = hashlib.shake_256(str_args.encode()).hexdigest(5)
        return hash



    def get_data_props(self):

        data_props = {}
        data_props['first_step'] = {}
        data_props['second_step'] = {}
        data_props['statistics'] = {}
        data_props['look_ups'] = {}
        data_props['final_data'] = {}

        data_props['look_ups']['dummy_col_dict'] = self.dummy_col_dict
        data_props['look_ups']['iter_dict'] = self.iter_dict
        data_props['look_ups']['iter_idx'] = self.iter_idx
        data_props['look_ups']['companies'] = self.companies
        data_props['look_ups']['out_lookup_col_number'] = self.latest_out['columns']
        data_props['look_ups']['out_lookup_col_name'] = self.latest_out['columns_lookup']

        data_props['first_step']['dataset'] = self.dataset
        data_props['first_step']['dataset_y_col'] = self.dataset_y_col
        data_props['first_step']['dataset_iter_col'] = self.dataset_iter_col
        data_props['first_step']['dataset_company_col'] = self.dataset_company_col
        data_props['first_step']['dataset_category_cols'] = self.dataset_category_cols
        data_props['first_step']['dataset_date_cols'] = self.dataset_date_cols

        data_props['first_step']['window_input_width'] = self.window_input_width
        data_props['first_step']['window_pred_width'] = self.window_pred_width
        data_props['first_step']['window_shift'] = self.window_shift

        data_props['second_step']['split_method'] = self.split_method
        data_props['second_step']['split_props'] = self.split_props

        data_props['second_step']['normalize_method'] = self.normalize_method
        data_props['second_step']['norm_keep_raw'] = self.norm_keep_raw

        data_props['second_step']['cols_drop'] = self.cols_drop
        data_props['second_step']['cols_just_these'] = self.cols_just_these
        data_props['second_step']['comps_exclude'] = self.comps_exclude
        data_props['second_step']['comps_just_these'] = self.comps_just_these

        data_props['second_step']['lagged_col_dict'] = self.lagged_col_dict

        data_props['final_data']['idx'] = {'train': self.latest_out['train']['idx'],
                                           'val': self.latest_out['val']['idx'],
                                           'test': self.latest_out['test']['idx']}
        data_props['final_data']['cols'] = {'X': list(self.latest_out['columns']['X'].values()),
                                            'y': list(self.latest_out['columns']['y'].values()),
                                            'lookup': {'col_number': data_props['look_ups']['out_lookup_col_number'],
                                                       'col_name': data_props['look_ups']['out_lookup_col_name']}}

        data_props['first_step']['data_hash'] = self.data_first_step_hash
        data_props['second_step']['data_hash'] = self.data_second_step_hash
        data_props['first_step_data_hash'] = data_props['first_step']['data_hash']
        data_props['second_step_data_hash'] = data_props['second_step']['data_hash']
        data_props['final_data']['data_hash_first_step'] = data_props['first_step']['data_hash']
        data_props['final_data']['data_hash_second_step'] = data_props['second_step']['data_hash']

        data_props['iter_step'] = self.latest_out['iter_step']

        for i in ['train', 'val', 'test']:
            tmp = self.latest_out[i]
            data_props['statistics'][i] = {}
            data_props['statistics'][i]['samples'] = tmp['X'].shape[0]
            data_props['statistics'][i]['companies'] = len(np.unique(tmp['idx'][:, 3]))
            data_props['statistics'][i]['features'] = tmp['X'].shape[-1]
            data_props['statistics'][i]['time_steps'] = len(np.unique(tmp['idx'][:, 0]))
            data_props['statistics'][i]['time_min'] = tmp['idx'][:, 1].astype(int).min()
            data_props['statistics'][i]['time_max'] = tmp['idx'][:, 2].astype(int).max()


        return data_props




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

            print('\nDumping cache to file for next use... (takes a bit because huge file and is being compressed)')
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
        self.data_first_step_hash = cache_hash
        self.computed = True

        print('Data class skeleton constructed (computed)! Ready to iterate across or subscript...')



    def _prep_final_dataset(self, df_or_args, norm_key=None, lower_idx=None, upper_idx=None, comp=None, norm_method=None):
        if type(df_or_args) != list:
            args = [(df_or_args, norm_key, lower_idx, upper_idx, comp, norm_method)]
        else:
            args = df_or_args

        X = []
        y = []
        idx = []
        warning = []

        final_cols = False
        last_norm_key = False
        last_comp = False

        s = 0
        ns = 0
        t = 0

        for df, norm_key, lower_idx, upper_idx, comp, norm_method in args:
            t += 1
            #print(type(norm_key), norm_key, type(lower_idx), lower_idx, type(upper_idx), upper_idx, type(comp), comp, type(norm_method), norm_method)
            tmp_df = df[(df.index >= int(lower_idx)) & (df.index <= int(upper_idx)) & (df[self.dataset_company_col] == int(comp))]


            if len(tmp_df) > 0:
                df = tmp_df.copy()

                if norm_method == 'block':
                    if last_norm_key is False:
                        mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, '__all__', 'mean').fillna(0)
                        std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, '__all__', 'std').fillna(1)
                        last_norm_key = True
                elif norm_method == 'time-step':
                    if norm_key != last_norm_key:
                        mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, '__all__', 'mean').fillna(0)
                        std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, '__all__', 'std').fillna(1)
                        last_norm_key = norm_key
                elif norm_method == 'set':
                    if norm_key != last_norm_key or comp != last_comp:
                        mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, f'c_{comp}', 'mean').fillna(0)
                        std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, f'c_{comp}', 'std').fillna(1)
                        last_norm_key = norm_key
                        last_comp = comp


                sort_cols = self.dataset_iter_col + [self.dataset_company_col]
                df.sort_values(sort_cols, inplace=True)

                if len(df) > (self.window_input_width + self.window_pred_width):
                    warning.append([comp, lower_idx, upper_idx, (f'Company {comp} has data duplicates in time_step {lower_idx}-{upper_idx}')])
                    #print(warning[-1])
                    df = df.drop_duplicates(subset=sort_cols, keep='last')
                    ns +=1
                if len(df) != (self.window_input_width + self.window_pred_width):
                    warning.append([comp, lower_idx, upper_idx, (f'Company {comp} has too few time points in time_step {lower_idx}-{upper_idx}')])
                    #print(warning[-1])
                    ns+=1
                else:

                    if final_cols == False:
                        cols = self.cols
                        if self.cols_just_these == False:
                            final_cols = [i for i in cols if (i not in self.cols_drop) and (i not in self.dataset_iter_col) and (i != self.dataset_company_col) and (i not in self.dataset_date_cols)]
                        else:
                            if type(self.cols_just_these) != list:
                                raise Exception('cols_just_these has to be a list of column names')
                            final_cols = [i for i in self.cols_just_these if (i not in self.dataset_iter_col) and (i != self.dataset_company_col) and (i not in self.dataset_date_cols)]

                        norm_cols = [i for i in df.columns.tolist() if (i in final_cols) and (i not in self.norm_keep_raw)]

                        for key, value in self.dummy_col_dict.items():
                            try:
                                final_cols.remove(key)
                                final_cols.extend(value)
                            except:
                                pass


                    df[norm_cols] = ((df[norm_cols] - mean[norm_cols].fillna(0)) / std[norm_cols].fillna(1)).replace([np.nan, np.inf, -np.inf], 0)

                    df = df.fillna(0).replace([np.nan, np.inf, -np.inf], 0)

                    X.append(pd.DataFrame(df.iloc[:-self.window_pred_width][final_cols]).values)
                    y.append(pd.DataFrame(df.iloc[-self.window_pred_width:][self.dataset_y_col]).values)
                    idx.append([norm_key, lower_idx, upper_idx, comp])
                    s+=1
            else:
                warning.append([comp, lower_idx, upper_idx, (f'Company {comp} has no data in time_step {lower_idx}-{upper_idx}')])
                #print(warning[-1])
                ns+=1

        #print(f'DONE! From {t} {s} were successful and {ns} not')

        return X, y, idx, [final_cols], warning



    def _final_dataset(self, train_dict, val_dict, test_dict, iter_step, data_hash):

        final_data_cache_folder = os.path.join(my.get_project_directories(key='cache_dir'), data_hash)
        if not os.path.exists(final_data_cache_folder):
            os.makedirs(final_data_cache_folder)

        column_hash = self._get_data_hash(self.cols_drop, self.dataset_date_cols, self.cols_just_these, self.norm_keep_raw, self.comps_just_these, self.normalize_method, self.comps_exclude)
        final_file = os.path.join(final_data_cache_folder, f'{column_hash}_{iter_step}_iter-step.npz')
        self.data_second_step_hash = column_hash

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

            start = time.time()

            ##################### TRAIN #####################

            train_todo_list = []
            norm_level = self.normalize_method


            for lower, upper in train_dict:
                norm_key = f't_{lower}_{upper}'
                comp_iter_list = my.custom_hdf5.get_comp_list(file=self.norm_param_file, norm_key=norm_key)
                if self.comps_just_these == False:
                    comp_iter_list = [i for i in comp_iter_list if (i not in self.comps_exclude) and (i != '__all__')]
                else:
                    if type(self.comps_just_these) != list:
                        raise Exception('comp_iter_list has to be a list of column names')
                    comp_iter_list = self.comps_just_these
                for comp in comp_iter_list:
                    comp = type(self.data[self.dataset_company_col].iloc[0])(comp[2:])
                    train_todo_list.append((self.data, norm_key, lower, upper, comp, norm_level))

            #train_todo_list = my.sort_list_of_sub(train_todo_list, sort_element=4)
            n = 750
            train_todo_list_of_list = [train_todo_list[i:i + n] for i in range(0, len(train_todo_list), n)]

            print(f'\nCaching, normalizing, and preparing train data for iteration-step/subscript {iter_step}:')
            train_X, train_y, train_idx, train_col_list, train_warning_list = my.multiprocessing_func_with_progressbar(func=self._prep_final_dataset, argument_list=train_todo_list_of_list, num_processes=-1, results='extend')
            """
            train_X, train_y, train_idx, train_col_list, train_warning_list = [], [], [], [], []
            i = 0
            for args in train_todo_list:
                train_X_tmp, train_y_tmp, train_idx_tmp, train_col_list_tmp, train_warning_list_tmp = self._prep_final_dataset(*args)
                train_X.append(train_X_tmp)
                train_y.append(train_y_tmp)
                train_idx.append(train_idx_tmp)
                train_col_list.append(train_col_list_tmp)
                train_warning_list.append(train_warning_list_tmp)
                if i > 30:
                    print(100)
                    i = 0
                i += 1
            """

            # ToDo: There are entire datasets with 500 tasks none resulting in long enough data
            train_col_list = list(filter(None, train_col_list))

            train_X = np.asarray(train_X)
            train_y = np.asarray(train_y)
            train_idx = np.asarray(train_idx)
            with open('train.txt', 'w') as f:
                for i in train_col_list:
                    f.write(str(i) + '\n')
            if all(elem == train_col_list[0] for elem in train_col_list) is False:
                raise Exception('Not all parts in train have the same columns!')



            ##################### VAL #####################

            val_todo_list = []

            for lower, upper in val_dict:
                norm_key = f't_{lower}_{upper}'
                comp_iter_list = my.custom_hdf5.get_comp_list(file=self.norm_param_file, norm_key=norm_key)
                if self.comps_just_these == False:
                    comp_iter_list = [i for i in comp_iter_list if (i not in self.comps_exclude) and (i != '__all__')]
                else:
                    if type(self.comps_just_these) != list:
                        raise Exception('comp_iter_list has to be a list of column names')
                    comp_iter_list = self.comps_just_these
                for comp in comp_iter_list:
                    comp = type(self.data[self.dataset_company_col].iloc[0])(comp[2:])
                    val_todo_list.append((self.data, norm_key, lower, upper, comp, norm_level))

            #val_todo_list = my.sort_list_of_sub(val_todo_list, sort_element=4)
            n_min = min(n, int(len(val_todo_list) / 7))
            val_todo_list_of_list = [val_todo_list[i:i + n_min] for i in range(0, len(val_todo_list), n_min)]

            print(f'\nCaching, normalizing, and preparing validation data for iteration-step/subscript {iter_step}:')
            val_X, val_y, val_idx, val_col_list, val_warning_list = my.multiprocessing_func_with_progressbar(func=self._prep_final_dataset, argument_list=val_todo_list_of_list, num_processes=-1, results='extend')

            val_col_list = list(filter(None, val_col_list))

            val_X = np.asarray(val_X)
            val_y = np.asarray(val_y)
            val_idx = np.asarray(val_idx)
            with open('val.txt', 'w') as f:
                for i in val_col_list:
                    f.write(str(i) + '\n')
            if all(elem == val_col_list[0] for elem in val_col_list) is False:
                raise Exception('Not all parts in validation have the same columns!')
            if val_col_list[0] != train_col_list[0]:
                raise Exception('Columns in validation are not equalt to train.')


            ##################### TEST #####################

            test_todo_list = []

            for lower, upper in test_dict:
                norm_key = f't_{lower}_{upper}'
                comp_iter_list = my.custom_hdf5.get_comp_list(file=self.norm_param_file, norm_key=norm_key)
                if self.comps_just_these == False:
                    comp_iter_list = [i for i in comp_iter_list if (i not in self.comps_exclude) and (i != '__all__')]
                else:
                    if type(self.comps_just_these) != list:
                        raise Exception('comp_iter_list has to be a list of column names')
                    comp_iter_list = self.comps_just_these
                for comp in comp_iter_list:
                    comp = type(self.data[self.dataset_company_col].iloc[0])(comp[2:])
                    test_todo_list.append((self.data, norm_key, lower, upper, comp, norm_level))

            #test_todo_list = my.sort_list_of_sub(test_todo_list, sort_element=4)
            n_min = min(n, int(len(test_todo_list)/7))
            test_todo_list_of_list = [test_todo_list[i:i + n_min] for i in range(0, len(test_todo_list), n_min)]

            print(f'\nCaching, normalizing, and preparing test data for iteration-step/subscript {iter_step}:')
            test_X, test_y, test_idx, test_col_list, test_warning_list = my.multiprocessing_func_with_progressbar(func=self._prep_final_dataset, argument_list=test_todo_list_of_list, num_processes=-1, results='extend')

            test_col_list = list(filter(None, test_col_list))

            test_X = np.asarray(test_X)
            test_y = np.asarray(test_y)
            test_idx = np.asarray(test_idx)
            with open('test.txt', 'w') as f:
                for i in test_col_list:
                    f.write(str(i) + '\n')
            if all(elem == test_col_list[0] for elem in test_col_list) is False:
                raise Exception('Not all parts in validation have the same columns!')
            if test_col_list[0] != train_col_list[0]:
                raise Exception('Columns in test are not equal to train and validation.')

            warning_list = train_warning_list + val_warning_list + test_warning_list

            ndarray_columns = {'X': dict(zip(range(len(train_col_list[0])), train_col_list[0])),
                               'y': dict(zip(range(len(self.dataset_y_col)), self.dataset_y_col))}
            pickle.dump(ndarray_columns, open((final_file[:-4] + '_cols.pkl'), 'wb'))
            warning_list = pd.DataFrame(warning_list, columns=["Company", "Lower IDX", "Upper IDX", "Message"])
            warning_list.drop_duplicates(keep='first', inplace=True, ignore_index=True)
            warning_list.to_csv((final_file[:-4] + '_warnings.csv'), index=False)

            np.savez_compressed(final_file, train_X=train_X, train_y=train_y, train_idx=train_idx, val_X=val_X, val_y=val_y, val_idx=val_idx, test_X=test_X, test_y=test_y, test_idx=test_idx)

            end = time.time()

            print(f'\nData cached for iteration-step {iter_step}. Took {int((end-start)/60)} min.')


        OUT = {'iter_step': iter_step,
               'train': {'X': train_X, 'y': train_y, 'idx': train_idx},
               'val':   {'X': val_X,   'y': val_y,   'idx': val_idx},
               'test':  {'X': test_X,  'y': test_y,  'idx': test_idx},
               'columns': ndarray_columns,
               'columns_lookup': {'X': dict(zip(list(ndarray_columns['X'].values()), list(ndarray_columns['X'].keys()))),
                                  'y': dict(zip(list(ndarray_columns['y'].values()), list(ndarray_columns['y'].keys())))}}
        self.latest_out = OUT
        return OUT


    def get_examples(self, out=None, example_list=[], example_len=5, random_seed=42):
        if out is None:
            out = self.latest_out
        if len(example_list) == 0:
            len_samples = len(out['test']['idx'])
            np.random.seed(random_seed)
            example_list = np.random.randint(0, len_samples, size=example_len).tolist()

        example_dict = {}
        example_dict['X'] = out['test']['X'][example_list, :, :]
        example_dict['X_ds'] = tf.data.Dataset.from_tensors(example_dict['X'])
        example_dict['y'] = out['test']['y'][example_list, :, :]
        y_col_idx_in_X = out['columns_lookup']['X'][self.dataset_y_col[0]]
        example_dict['y_hist'] = out['test']['X'][example_list, :, y_col_idx_in_X]
        y_col_idx_in_y = out['columns_lookup']['y'][self.dataset_y_col[0]]
        example_dict['y_true'] = out['test']['y'][example_list, :, y_col_idx_in_y]

        example_dict['columns'] = out['columns']
        example_dict['columns_lookup'] = out['columns_lookup']
        lower_idx = [int(i) for i in out['test']['idx'][example_list, 1]]
        upper_idx = [int(i) for i in out['test']['idx'][example_list, 2]]
        example_dict['time_step'] = out['test']['idx'][example_list, 0]
        example_dict['company'] = out['test']['idx'][example_list, -1]
        idx = []
        for l_idx, u_idx in zip(lower_idx, upper_idx):
            idx.append([i for i in self.iter_idx if i >= l_idx and i <= u_idx])
        example_dict['t_idx'] = idx

        example_dict['examples_num'] = example_len
        example_dict['examples_list'] = example_list

        norm_param = []
        for norm_key, comp in zip(example_dict['time_step'], example_dict['company']):
            if self.normalize_method == 'time-step':
                mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, '__all__', 'mean').fillna(0)
                std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, '__all__', 'std').fillna(1)
            elif self.normalize_method == 'set':
                mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, f'c_{comp}', 'mean').fillna(0)
                std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, norm_key, f'c_{comp}', 'std').fillna(1)
            elif self.normalize_method == 'block':
                mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, '__all__', 'mean').fillna(0)
                std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, '__all__', 'std').fillna(1)
            norm_param.append({'mean': mean, 'std': std})

        example_dict['norm_param'] = norm_param
        example_dict['y_cols'] = self.dataset_y_col

        return example_dict



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


    def tsds_dataset(self, out='all', out_dict=None, transpose_y=True):
        if out_dict is None:
            out_dict = self.latest_out
        if out=='all':
            out = ['train', 'val', 'test']
        if type(out) != list:
            out = list(out)
        output = []
        for i in out:
            tmp_y = out_dict[i]['y']
            if transpose_y:
                tmp_y = tmp_y.reshape((-1, tmp_y.shape[2], tmp_y.shape[1]))
            tmp = tf.data.Dataset.from_tensors((out_dict[i]['X'], tmp_y))
            output.append(tmp)
        return output



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

    def __help__(self):
        self.help()

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
            props_dict = pd.DataFrame.from_dict(dict(zip(range(1, len(tmp_idx) + 1), tmp_idx.tolist())), orient='index', columns=['period', self.dataset_company_col, 'lower_idx', 'upper_idx'])
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
    # ToDo: tfds data generator
    # ToDo: data example/graph
    # ToDo: shuffle data
    # ToDo: add lagged variables
    # ToDo: outlier normalization
    # ToDo: rolling block step size of iteration
    # ToDo: add block normalization


    data = data_prep(dataset='final_data_2', recache=False, keep_raw_cols='default', drop_cols='default')

    data.window(input_width=5*4, pred_width=4, shift=1)

    #data.block_static_split(val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=5*4*2, val_time_steps=1, test_time_steps=1, shuffle=True)
    #data.single_time_rolling(val_time_steps=1, test_time_steps=1, shuffle=True)

    # data.normalize(method='block')
    data.normalize(method='time')
    # data.normalize(method='set')

    data.compute()


    out = data['200201_201903']
    ds_train, ds_val, ds_test = data.tsds_dataset(out='all', out_dict=None)
    #data.export_to_excel()

    print(data)


    #data.feature_filter(include_features_list=None, exclude_features_list=None)
    #data.company_filter(include_comp_list=None, exclude_comp_list=None)

    #print(train_ds, len(train_ds))





