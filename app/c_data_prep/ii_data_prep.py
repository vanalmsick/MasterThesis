import os, sys, zlib, hashlib, warnings, math, datetime, random
import psycopg2, psycopg2.extras, pickle, progressbar, time, hashlib
import pandas as pd
import numpy as np
import tensorflow as tf

# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')
from app.z_helpers import helpers as my




class data_prep:
    def __init__(self, dataset, iter_cols=None, comp_col=None, y_cols=None, recache=False, category_cols=[], date_cols=[], keep_raw_cols=[], drop_cols=[]):
        self.recache = recache


        if type(dataset) == str:

            from app.b_data_cleaning import get_dataset_registry
            dataset_props = get_dataset_registry()[dataset]

            comp_col = dataset_props['company_col']
            iter_cols = dataset_props['iter_cols']
            industry_col = dataset_props['industry_col']
            category_cols = dataset_props['category_cols'] if 'category_cols' in dataset_props and category_cols == [] else category_cols
            date_cols = dataset_props['date_cols'] if 'date_cols' in dataset_props and date_cols == [] else date_cols
            y_cols = dataset_props['y_col'] if 'y_col' in dataset_props and y_cols is False else y_cols

            from app.c_data_prep.i_feature_engineering import feature_engerneeing
            dataset = feature_engerneeing(dataset=dataset_name, comp_col=comp_col, time_cols=iter_cols, industry_col=industry_col, recache=recache)

        if iter_cols is None or comp_col is None or y_cols is None:
            raise Exception(f'One of them is None which is not allowed: iter_cols = {iter_cols}, comp_col = {comp_col}, y_cols = {y_cols}')

        from pandas.util import hash_pandas_object
        dataset_hash = hash_pandas_object(dataset).sum()
        self.dataset = dataset_hash

        self.dataset_iter_col = iter_cols
        self.dataset_company_col = comp_col
        self.dataset_y_col = y_cols
        self.dataset_category_cols = category_cols
        self.dataset_date_cols = date_cols


        self.keep_raw_cols = keep_raw_cols

        self.cols_drop = drop_cols
        self.cols_just_these = False
        self.y_drop = []
        self.y_just_these = False

        self.comps_exclude = []
        self.comps_just_these = False



        # Format the data to the correct format
        for cat in self.dataset_category_cols:
            dataset[cat] = dataset[cat].astype('category')
        for dt_col in self.dataset_date_cols:
            dataset[dt_col] = pd.to_datetime(dataset[dt_col])



        i_col = pd.DataFrame()
        for col in self.dataset_iter_col:
            i_col[col] = dataset[col].astype(int).astype(str).str.zfill(2)
        dataset['iter_col'] = i_col.agg(''.join, axis=1).astype(int)
        self.iter_idx = sorted(np.array(dataset['iter_col'].unique()).tolist())
        dataset.set_index(keys='iter_col', inplace=True)
        dataset.sort_index(inplace=True)




        self.raw_data = dataset.copy()
        self.mod_data = dataset.copy()
        self.computed = False




    def _transform_dummy_vars(self):
        cat_cols = self.raw_data.select_dtypes(include=['object', 'category']).columns.tolist()
        # ToDo: remove dummy variable transformation as done before
        cat_cols = [i for i in cat_cols if i != self.dataset_company_col]
        if 'industry' in self.cols:
            self.cols.remove('industry')
        df = self.raw_data.copy()
        dummy_dict = {}
        #for col in cat_cols:
        #    tmp = pd.get_dummies(df[col], dummy_na=True, drop_first=True, prefix=str(col))
        #    dummy_dict[col] = tmp.columns.tolist()
        #    df = pd.concat([df, tmp], axis=1)
        #self.data = df.drop(columns=cat_cols)
        self.data = df
        # ToDo: Implement real NaN handling
        #self.data = self.data.fillna(self.data.mean().fillna(0))
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
        tmp_mean = tmp_df[norm_cols].fillna(tmp_df[norm_cols].mean()).mean()
        tmp_std = tmp_df[norm_cols].fillna(tmp_df[norm_cols].mean()).std()
        tmp_mean_np = my.custom_hdf5.pd_series_to_2d_array(pd_series=tmp_mean)
        tmp_std_np = my.custom_hdf5.pd_series_to_2d_array(pd_series=tmp_std)
        norm_param['__all__'] = {'mean': tmp_mean_np, 'std': tmp_std_np}

        for comp in comp_list:
            tmp_df = df[(df[self.dataset_company_col] == comp) & (df.index >= idx_lower) & (df.index <= idx_upper)]
            tmp_mean = tmp_df[norm_cols].fillna(tmp_df[norm_cols].mean()).mean()
            tmp_std = tmp_df[norm_cols].fillna(tmp_df[norm_cols].mean()).std()
            tmp_mean_np = my.custom_hdf5.pd_series_to_2d_array(pd_series=tmp_mean)
            tmp_std_np = my.custom_hdf5.pd_series_to_2d_array(pd_series=tmp_std)
            norm_param[f'c_{comp}'] = {'mean': tmp_mean_np, 'std': tmp_std_np}

        return norm_param

    def _get_normalization_param(self, df, idx_dict):
        # ToDo: Check if Normalization is OOS
        print('\nCaching indices/iterators and normalization parameters:')
        norm_get__all__ = {}
        norm_get = {}
        for all_key, time_dict in idx_dict.items():
            for i, j in time_dict['train']:
                norm_get[f't_{i}_{j}'] = (i, j)
                # Additional indices for block normalization
                if ('t_' + all_key) not in norm_get__all__:
                    norm_get__all__[('t_' + all_key)] = (i, j)
                if norm_get__all__[('t_' + all_key)][0] > i:
                    old_j = norm_get__all__[('t_' + all_key)][1]
                    norm_get__all__[('t_' + all_key)] = (i, old_j)
                if norm_get__all__[('t_' + all_key)][1] < i:
                    old_i = norm_get__all__[('t_' + all_key)][0]
                    norm_get__all__[('t_' + all_key)] = (old_i, j)
            for i, j in time_dict['val']:
                norm_get[f't_{i}_{j}'] = (i, j)
            for i, j in time_dict['test']:
                norm_get[f't_{i}_{j}'] = (i, j)



        input_data_pairs = []
        for key, idx in norm_get.items():
            input_data_pairs.append((df, idx[0], idx[1]))

        all__input_data_pairs = []
        for key, idx in norm_get__all__.items():
            all__input_data_pairs.append((df, idx[0], idx[1]))

        data = my.multiprocessing_func_with_progressbar(func=self._normalization_multicore_function, argument_list=input_data_pairs, num_processes=-1)
        norm_param_dict = dict(zip(list(norm_get.keys()), data))

        print('\nCaching block normalization parameters:')

        all__data = my.multiprocessing_func_with_progressbar(func=self._normalization_multicore_function, argument_list=all__input_data_pairs, num_processes=-1)
        all__norm_param_dict = dict(zip(list(norm_get__all__.keys()), all__data))

        norm_param_dict['__all__'] = all__norm_param_dict

        return norm_param_dict




    def _get_data_hash(self, *args):
        str_args = (str(args)[1:-1]).replace("'", "").replace(", ", "/")
        hash = hashlib.shake_256(str_args.encode()).hexdigest(5)
        return hash



    def get_data_props(self):

        data_props = {}
        data_props['first_step'] = {}
        data_props['second_step'] = {}
        data_props['third_filter'] = {}
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

        data_props['third_filter']['cols_drop'] = self.cols_drop
        data_props['third_filter']['cols_just_these'] = self.cols_just_these
        data_props['third_filter']['y_cols_drop'] = self.y_drop
        data_props['third_filter']['y_cols_just_these'] = self.y_just_these
        data_props['third_filter']['comps_exclude'] = self.comps_exclude
        data_props['third_filter']['comps_just_these'] = self.comps_just_these

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



        cache_hash = self._get_data_hash(self.dataset, self.data.index.unique().tolist(), self.window_input_width, self.window_pred_width, self.window_shift, hist_periods_in_block, val_time_steps, test_time_steps)
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



    def _prep_final_dataset(self, df_or_args, norm_key=None, lower_idx=None, upper_idx=None, comp=None, norm_method=None, iter_step=None):
        if type(df_or_args) != list:
            args = [(df_or_args, norm_key, lower_idx, upper_idx, comp, norm_method, iter_step)]
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

        for df, norm_key, lower_idx, upper_idx, comp, norm_method, iter_step in args:
            t += 1
            #print(type(norm_key), norm_key, type(lower_idx), lower_idx, type(upper_idx), upper_idx, type(comp), comp, type(norm_method), norm_method)
            tmp_df = df[(df.index >= int(lower_idx)) & (df.index <= int(upper_idx)) & (df[self.dataset_company_col] == comp)]


            if len(tmp_df) > 0:
                df = tmp_df.copy()

                if norm_method == 'block':
                    if last_norm_key is False:
                        mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, '__all__', f't_{iter_step}', '__all__', 'mean').fillna(0)
                        std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, '__all__', f't_{iter_step}', '__all__', 'std').fillna(1)
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
                elif norm_method == 'no':
                    cols = df.columns.tolist()
                    mean = pd.Series(0, index=cols)
                    std = pd.Series(1, index=cols)


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
                        final_cols = [i for i in cols if (i not in self.dataset_iter_col) and (i != self.dataset_company_col) and (i not in self.dataset_date_cols)]

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

        column_hash = self._get_data_hash(self.dataset_date_cols, self.norm_keep_raw, self.normalize_method, self.dataset_y_col)
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
            self.raw_data = pd.read_pickle(final_file[:-4] + '_y.pkl')
            #y_data = pickle.load(open((final_file[:-4] + '_y.pkl'), 'rb'))


            print('Got data from cached file.')

        else:

            start = time.time()

            y_data = {}


            ##################### TRAIN #####################

            train_todo_list = []
            norm_level = self.normalize_method


            for lower, upper in train_dict:
                norm_key = f't_{lower}_{upper}'
                comp_iter_list = my.custom_hdf5.get_comp_list(file=self.norm_param_file, norm_key=norm_key)
                comp_iter_list = [i for i in comp_iter_list if (i != '__all__')]
                for comp in comp_iter_list:
                    comp = type(self.data[self.dataset_company_col].iloc[0])(comp[2:])
                    train_todo_list.append((self.data, norm_key, lower, upper, comp, norm_level, iter_step))

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
            #with open('train.txt', 'w') as f:
            #    for i in train_col_list:
            #        f.write(str(i) + '\n')
            if all(elem == train_col_list[0] for elem in train_col_list) is False:
                raise Exception('Not all parts in train have the same columns!')



            ##################### VAL #####################

            val_todo_list = []

            for lower, upper in val_dict:
                norm_key = f't_{lower}_{upper}'
                comp_iter_list = my.custom_hdf5.get_comp_list(file=self.norm_param_file, norm_key=norm_key)
                comp_iter_list = [i for i in comp_iter_list if (i != '__all__')]
                for comp in comp_iter_list:
                    comp = type(self.data[self.dataset_company_col].iloc[0])(comp[2:])
                    val_todo_list.append((self.data, norm_key, lower, upper, comp, norm_level, iter_step))

            #val_todo_list = my.sort_list_of_sub(val_todo_list, sort_element=4)
            n_min = min(n, int(len(val_todo_list) / 7))
            val_todo_list_of_list = [val_todo_list[i:i + n_min] for i in range(0, len(val_todo_list), n_min)]

            print(f'\nCaching, normalizing, and preparing validation data for iteration-step/subscript {iter_step}:')
            val_X, val_y, val_idx, val_col_list, val_warning_list = my.multiprocessing_func_with_progressbar(func=self._prep_final_dataset, argument_list=val_todo_list_of_list, num_processes=-1, results='extend')

            val_col_list = list(filter(None, val_col_list))

            val_X = np.asarray(val_X)
            val_y = np.asarray(val_y)
            val_idx = np.asarray(val_idx)
            #with open('val.txt', 'w') as f:
            #    for i in val_col_list:
            #        f.write(str(i) + '\n')
            if all(elem == val_col_list[0] for elem in val_col_list) is False:
                raise Exception('Not all parts in validation have the same columns!')
            if val_col_list[0] != train_col_list[0]:
                raise Exception('Columns in validation are not equalt to train.')


            ##################### TEST #####################

            test_todo_list = []

            for lower, upper in test_dict:
                norm_key = f't_{lower}_{upper}'
                comp_iter_list = my.custom_hdf5.get_comp_list(file=self.norm_param_file, norm_key=norm_key)
                comp_iter_list = [i for i in comp_iter_list if (i != '__all__')]
                for comp in comp_iter_list:
                    comp = type(self.data[self.dataset_company_col].iloc[0])(comp[2:])
                    test_todo_list.append((self.data, norm_key, lower, upper, comp, norm_level, iter_step))

            #test_todo_list = my.sort_list_of_sub(test_todo_list, sort_element=4)
            n_min = min(n, int(len(test_todo_list)/7))
            test_todo_list_of_list = [test_todo_list[i:i + n_min] for i in range(0, len(test_todo_list), n_min)]

            print(f'\nCaching, normalizing, and preparing test data for iteration-step/subscript {iter_step}:')
            test_X, test_y, test_idx, test_col_list, test_warning_list = my.multiprocessing_func_with_progressbar(func=self._prep_final_dataset, argument_list=test_todo_list_of_list, num_processes=-1, results='extend')

            test_col_list = list(filter(None, test_col_list))

            test_X = np.asarray(test_X)
            test_y = np.asarray(test_y)
            test_idx = np.asarray(test_idx)
            #with open('test.txt', 'w') as f:
            #    for i in test_col_list:
            #        f.write(str(i) + '\n')
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
            #pickle.dump(y_data, open((final_file[:-4] + '_y.pkl'), 'wb'))
            self.raw_data.to_pickle(final_file[:-4] + '_y.pkl')

            end = time.time()

            print(f'\nData cached for iteration-step {iter_step}. Took {int((end-start)/60)} min.')


        OUT = {'iter_step': iter_step,
               'train': {'X': train_X, 'y': train_y, 'idx': train_idx},
               'val':   {'X': val_X,   'y': val_y,   'idx': val_idx},
               'test':  {'X': test_X,  'y': test_y,  'idx': test_idx},
               'raw_data': self.raw_data,
               'columns': ndarray_columns,
               'columns_lookup': {'X': dict(zip(list(ndarray_columns['X'].values()), list(ndarray_columns['X'].keys()))),
                                  'y': dict(zip(list(ndarray_columns['y'].values()), list(ndarray_columns['y'].keys())))}}

        OUT = self._apply_filters(OUT)

        self.latest_out = OUT
        return OUT


    def get_examples(self, out=None, example_list=[], y_col='y_eps', example_len=5, random_seed=42):
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
        y_col_idx_in_X = list(out['columns_lookup']['X'].values())[0]
        example_dict['y_hist'] = out['test']['X'][example_list, :, y_col_idx_in_X]
        y_col_idx_in_y = out['columns_lookup']['y'][y_col]
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
                latest_block = 't_' + self.latest_out['iter_step']
                mean = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, '__all__', latest_block, '__all__', 'mean').fillna(0)
                std = my.custom_hdf5.hdf5_to_pd(self.norm_param_file, '__all__', latest_block, '__all__', 'std').fillna(1)
            elif self.normalize_method == 'no':
                mean = 0
                std = 1
            norm_param.append({'mean': mean, 'std': std})

        example_dict['norm_param'] = norm_param
        example_dict['y_cols'] = self.dataset_y_col

        return example_dict



    ############# ITERABLE #############

    def __iter__(self):
        if self.computed is False:
            self.compute()
        self._custom_iter_ = iter(list(self.iter_dict.keys()))
        return self

    def __next__(self):
        current = next(self._custom_iter_)
        data_dict = self._final_dataset(train_dict=self.iter_dict[current]['train'], val_dict=self.iter_dict[current]['val'],
                                        test_dict=self.iter_dict[current]['test'], iter_step=current, data_hash=self.data_hash)
        return data_dict

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
        self.latest_out = out_dict
        if out=='all':
            out = ['train', 'val', 'test']
        if type(out) != list:
            out = list(out)
        output = []
        for i in out:
            tmp_y = out_dict[i]['y']
            if transpose_y:
                tmp_y = tmp_y.reshape((-1, tmp_y.shape[2], tmp_y.shape[1]))
            tmp = tf.data.Dataset.from_tensors((out_dict[i]['X'].astype(np.float32), tmp_y.astype(np.float32)))
            output.append(tmp)
        return output

    def np_dataset(self, out='all', out_dict=None, transpose_y=True):
        if out_dict is None:
            out_dict = self.latest_out
        self.latest_out = out_dict
        if out=='all':
            out = ['train', 'val', 'test']
        if type(out) != list:
            out = list(out)
        output = []
        for i in out:
            tmp_y = out_dict[i]['y']
            if transpose_y:
                tmp_y = tmp_y.reshape((-1, tmp_y.shape[2], tmp_y.shape[1]))
            tmp = (out_dict[i]['X'], tmp_y)
            output.append(tmp)
        return output

    def y_dataset(self, out='all', out_dict=None):
        if out_dict is None:
            out_dict = self.latest_out
        self.latest_out = out_dict

        raw_data = out_dict['raw_data']

        arima_df = raw_data.pivot(index=None, columns=self.dataset_company_col, values=self.y_just_these[0])
        idx = self.iter_dict[out_dict['iter_step']]

        arima_df = arima_df.copy()
        arima_df = arima_df.loc[idx['train'][0][0]:idx['test'][-1][-1]]
        arima_df = arima_df.dropna(axis=1)

        out_dict = {'train': {'lower': idx['train'][0][0], 'upper': idx['train'][-1][-1]},
                    'val': {'lower': min([i for i in arima_df.index.tolist() if i > idx['train'][-1][-1] and i <= idx['val'][-1][-1]]), 'upper': idx['val'][-1][-1]},
                    'test': {'lower': min([i for i in arima_df.index.tolist() if i > idx['val'][-1][-1] and i <= idx['test'][-1][-1]]), 'upper': idx['test'][-1][-1]}}

        for set in ['train', 'val', 'test']:
            i_lower = out_dict[set]['lower']
            i_upper = out_dict[set]['upper']
            tmp = arima_df.loc[i_lower:i_upper].copy()
            #tmp = tmp.dropna(axis=1)
            out_dict[set] = {'y_data': tmp.values.T.astype(float),
                             'comps': tmp.columns.tolist(),
                             'iter_steps': tmp.index.tolist()}

        return out_dict

    def df_dataset(self, out='all', year_idx=-1, out_dict=None, transpose_y=True):
        if out_dict is None:
            out_dict = self.latest_out
        self.latest_out = out_dict
        if out=='all':
            out = ['train', 'val', 'test']
        if type(out) != list:
            out = list(out)
        output = []
        for i in out:
            tmp_y = out_dict[i]['y']
            if transpose_y:
                tmp_y = tmp_y.reshape((-1, tmp_y.shape[2], tmp_y.shape[1]))
            tmp_y = tmp_y[:, year_idx, :]
            tmp_X = out_dict[i]['X'][:, year_idx, :]
            X_cols = list(out_dict['columns']['X'].values())
            tmp_X = pd.DataFrame(tmp_X, columns=X_cols)
            tmp = (tmp_X, tmp_y)
            output.append(tmp)
        return output



    def __str__(self):
        return f"\n\nCustom-Data class:\n" \
               "-------------------------------------------------------------\n" \
               f"Dataset: {self.dataset}\n" \
               f"Recache: {self.recache}\n" \
               f"Companies ({len(self.companies)}): {str(self.companies)}\n" \
               f"Time-steps ({len(self.iter_dict)}): {str(list(self.iter_dict.keys()))}\n" \
               f"Normaliation method: {self.normalize_method}\n" \
               f"Split method: {self.split_method}\n" \
               f"Split props: {self.split_props}\n" \
               f"Window props: ({self.window_input_width}, {self.window_pred_width}, {self.window_shift})\n" \
               f"Data hash: {self.data_hash}\n"

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
        elif method == False or method == 'no':
            self.normalize_method = 'no'
        else:
            raise Exception(f'UNKNOWN method={method} for normalization. Possible are: block / time / set.')


    #############################################################
    ### After compute ###

    def filter_features(self, just_include=None, exclude=None):
        if just_include is not None:
            self.cols_just_these = just_include
        if exclude is not None:
            self.cols_drop = exclude

    def filter_columns(self, just_include=None, exclude=None):
        self.filter_features(just_include=just_include, exclude=exclude)

    def filter_y(self, just_include=None, exclude=None):
        if just_include is not None:
            self.y_just_these = just_include
        if exclude is not None:
            self.y_drop = exclude

    def filter_companies(self, just_include=None, exclude=None):
        if self.normalize_method == 'block':
            warnings.warn('You set a company filter but normalize across the entire block including all companies. The company filter is not applied to the normalization.')
            self.computed = False

        if just_include is not None:
            self.comps_just_these = just_include
        if exclude is not None:
            self.comps_exclude = exclude



    def _apply_filters(self, OUT):

        iter_step = OUT['iter_step']
        train_X = OUT['train']['X']
        train_y = OUT['train']['y']
        train_idx = OUT['train']['idx']
        val_X = OUT['val']['X']
        val_y = OUT['val']['y']
        val_idx = OUT['val']['idx']
        test_X = OUT['test']['X']
        test_y = OUT['test']['y']
        test_idx = OUT['test']['idx']
        cols_X = OUT['columns']['X']
        cols_y = OUT['columns']['y']
        cols_lookup_X = OUT['columns_lookup']['X']
        cols_lookup_y = OUT['columns_lookup']['y']
        raw_data = OUT['raw_data']


        # Filter X columns
        if self.cols_just_these != False or self.cols_drop != []:
            cols = pd.Series(OUT['columns']['X'])
            if self.cols_just_these != False:
                cols = cols[cols.isin(self.cols_just_these)]
            if self.cols_drop != []:
                cols = cols[~cols.isin(self.cols_drop)]
            i_cols = cols.index.tolist()
            cols.reset_index(drop=True, inplace=True)
            cols_X = cols.to_dict()
            cols_lookup_X = dict((v, k) for k, v in cols_X.items())

            train_X = train_X[:, :, i_cols]
            val_X = val_X[:, :, i_cols]
            test_X = test_X[:, :, i_cols]


        # Filter y columns
        if self.y_just_these != False or self.y_drop != []:
            cols = pd.Series(OUT['columns']['y'])
            if self.y_just_these != False:
                cols = cols[cols.isin(self.y_just_these)]
            if self.y_drop != []:
                cols = cols[~cols.isin(self.y_drop)]
            i_cols = cols.index.tolist()
            cols.reset_index(drop=True, inplace=True)
            cols_y = cols.to_dict()
            cols_lookup_y = dict((v, k) for k, v in cols_y.items())

            train_y = train_y[:, :, i_cols]
            val_y = val_y[:, :, i_cols]
            test_y = test_y[:, :, i_cols]



        def _sub_help_filter(X, y, idx):  # Helper for company filtering to not copy code 3x times for train val test
            comps = pd.Series(idx[:, -1])
            if self.comps_just_these != False:
                comps = comps[comps.isin(self.comps_just_these)]
            if self.comps_exclude != []:
                comps = comps[~comps.isin(self.comps_exclude)]
            i_comps = comps.index.tolist()

            X = X[i_comps, :, :]
            y = y[i_comps, :, :]
            idx = idx[i_comps, :]

            return X, y, idx


        # Filter companies
        if self.comps_just_these != False or self.comps_exclude != []:
            train_X, train_y, train_idx = _sub_help_filter(train_X, train_y, train_idx)
            val_X, val_y, val_idx = _sub_help_filter(val_X, val_y, val_idx)
            test_X, test_y, test_idx = _sub_help_filter(test_X, test_y, test_idx)
            warnings.warn('Implement filtering on y_dataset')



        OUT = {'iter_step': iter_step,
               'train': {'X': train_X, 'y': train_y, 'idx': train_idx},
               'val': {'X': val_X, 'y': val_y, 'idx': val_idx},
               'test': {'X': test_X, 'y': test_y, 'idx': test_idx},
               'raw_data': raw_data,
               'columns': {'X': cols_X, 'y': cols_y},
               'columns_lookup': {'X': cols_lookup_X, 'y': cols_lookup_y}}

        return OUT

    #############################################################


















if __name__ == '__main__':
    # ToDo: tfds data generator
    # ToDo: data example/graph
    # ToDo: shuffle data
    # ToDo: add lagged variables
    # ToDo: outlier normalization
    # ToDo: rolling block step size of iteration
    # ToDo: add block normalization

    # Working directory must be the higher .../app folder
    from app.z_helpers import helpers as my
    my.convenience_settings()

    # Dataset to use
    dataset_name = 'handpicked_dataset'


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

    df_to_use = feature_engerneeing(dataset=df_cleaned, comp_col=comp_col, time_cols=time_cols, industry_col=industry_col)

    y_cols = ['y_roe']
    data = data_prep(dataset=df_to_use, y_cols=y_cols, iter_cols=time_cols, comp_col=comp_col, keep_raw_cols=[], drop_cols=[])

    data.window(input_width=5*4, pred_width=4, shift=1)

    #data.block_static_split(val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True)
    data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=5*4*2, val_time_steps=4, test_time_steps=4, shuffle=True)
    #data.single_time_rolling(val_time_steps=1, test_time_steps=1, shuffle=True)

    # data.normalize(method='block')
    data.normalize(method='time')
    # data.normalize(method='set')

    data.compute()


    out = data['199904_201804']
    ds_train, ds_val, ds_test = data.tsds_dataset(out='all', out_dict=None)
    data.export_to_excel()

    print(data)


    #data.feature_filter(include_features_list=None, exclude_features_list=None)
    #data.company_filter(include_comp_list=None, exclude_comp_list=None)

    #print(train_ds, len(train_ds))





