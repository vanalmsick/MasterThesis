import os, sys, zlib, hashlib
import psycopg2, psycopg2.extras, pickle
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


def filter_func(features):
    # this doesn't work
    # result = features['label'] == 'some_label_value'
    # neither this
    label = tf.unstack(features['gvkey'])
    label = label[0]
    result = tf.reshape(tf.equal(label, 'some_label_value'), [])
    return result


class data:
    def __init__(self, dataset='final_data_2', recache=False):
        self.dataset, self.recache = dataset, recache

        df, self.raw_file_path = _download_data_from_sql(data_version=dataset, recache=recache)

        dataset_pops = {
            'final_data': {'iter_col': ['period_year', 'period_qrt'], 'company_col': 'gvkey', 'y_col': 'tr_f_ebit',
                           'category_cols': ['gsector'], 'date_cols': [], 'keep_raw_cols': [],
                           'drop_cols': ['ric', 'curcdq', 'rp', 'updq', 'iid', 'exchg', 'costat', 'fic', 'srcq',
                                         'curncdq', 'acctsdq', 'acctstdq', 'ggroup']},
            'final_data_2': {'iter_col': ['period_year', 'period_qrt'], 'company_col': 'gvkey', 'y_col': 'tr_f_ebit',
                             'category_cols': ['gsector'], 'date_cols': [], 'keep_raw_cols': [],
                             'drop_cols': ['ric', 'curcdq', 'rp', 'updq', 'iid', 'exchg', 'costat', 'fic', 'srcq',
                                           'curncdq', 'acctsdq', 'acctstdq', 'ggroup']}}

        self.props = dataset_pops[dataset]
        self._dopped_cols = self.props['iter_col'] + self.props['drop_cols']

        for cat in self.props['category_cols']:
            df[cat] = df[cat].astype('category')
        for dt_col in self.props['date_cols']:
            df[dt_col] = pd.to_datetime(df[dt_col])
        df[self.props['iter_col']] = df[self.props['iter_col']].astype(int)

        i_col = pd.DataFrame()
        for col in self.props['iter_col']:
            i_col[col] = df[col].astype(int).astype(str).str.zfill(2)
        df['iter_col'] = i_col.agg(''.join, axis=1).astype(int)
        self.iter_idx = np.array(df['iter_col'].unique()).tolist()
        df.set_index(keys='iter_col', inplace=True)
        df.sort_index(inplace=True)

        df.drop(columns=self._dopped_cols, inplace=True, errors='ignore')
        self.raw_data = df.copy()
        self.columns = df.columns.tolist()
        self.companies = np.array(df[self.props['company_col']].unique()).tolist()


    def _split_df_comp(self, df):
        # Split  by company
        companies = np.array(df[self.props['company_col']].unique()).tolist()
        comp_dict = {}
        for comp in companies:
            tmp_df = df[df[self.props['company_col']] == comp].sort_index()
            comp_dict[comp] = tmp_df[[i for i in self.columns if i != self.props['company_col']]]
        return comp_dict




    def train_val_test(self, multi_comp=True, val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True):
        self.multi_comp = multi_comp
        df = self.raw_data.copy()
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(self.companies)
        val_i, test_i = -int((val_comp_size + test_comp_size) * len(self.companies)), -int(test_comp_size * len(self.companies))
        val_comp_list, test_comp_list = self.companies[val_i:test_i], self.companies[test_i:]
        val_j, test_j = -int((val_time_size + test_time_size) * len(self.iter_idx)), -int(test_time_size * len(self.iter_idx))
        val_t, test_t = self.iter_idx[val_j], self.iter_idx[test_j]

        if multi_comp:
            self.val_df = df[(df[self.props['company_col']].isin(val_comp_list) | ((df.index < test_t) & (df.index > val_t)))]
            self.test_df = df[(df[self.props['company_col']].isin(test_comp_list) | (df.index >= test_t))]
            self.train_df = df[(~df[self.props['company_col']].isin(val_comp_list) & ~df[self.props['company_col']].isin(test_comp_list) & (df.index <= val_t))]
        else:
            self.val_df = df[((df.index < test_t) & (df.index > val_t))]
            self.test_df = df[(df.index >= test_t)]
            self.train_df = df[(df.index <= val_t)]

        print(
            'Total dataset size: {}\n{} samples for training ({}%) / {} samples for validation ({}%) / {} samples for testing ({}%)'.format(
                len(self.raw_data), len(self.train_df), int(len(self.train_df) / len(self.raw_data) * 100),
                len(self.val_df), int(len(self.val_df) / len(self.raw_data) * 100), len(self.test_df),
                int(len(self.test_df) / len(self.raw_data) * 100)))

        self.data_hash = str(hashlib.sha256(self.train_df.values.tobytes()).hexdigest()) + str(hashlib.sha256(self.val_df.values.tobytes()).hexdigest()) + str(hashlib.sha256(self.test_df.values.tobytes()).hexdigest())
        self.data_hash = hashlib.md5(self.data_hash.encode()).hexdigest()

        self.val_sorted = self._split_df_comp(df=self.val_df)
        self.test_sorted = self._split_df_comp(df=self.test_df)
        self.train_sorted = self._split_df_comp(df=self.train_df)




class WindowGenerator():
    def __init__(self, input_width, label_width, shift, train_df, val_df=None, test_df=None, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def _df_to_ds(self, df):
        data = np.array(df, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

def make_dataset(train_df, val_df, test_df, input_width, label_width, shift, label_columns, data_hash):
    data = {'train': train_df, 'val': val_df, 'test': test_df}
    first_run = True

    cache_folder = os.path.join(my.get_project_directories(key='cache_dir'), data_hash, (str(input_width) + '_' + str(label_width) + '_' + str(shift) + '_' + hashlib.md5(str(label_columns).encode()).hexdigest()))
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    if not os.path.exists(os.path.join(cache_folder, 'train.pkl')):

        for key in data:
            X = []
            y = []
            for one_comp_data in data[key]:
                df = data[key][one_comp_data]
                if first_run:
                    window = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift, train_df=df, label_columns=label_columns)
                    first_run = False
                if len(df) > 1:
                    ds = window._df_to_ds(df=df)
                    np_iter = ds.as_numpy_iterator()

                    for appX, appy in np_iter:
                        X.append(appX)
                        y.append(appy)

            X = np.concatenate(X, axis=0)
            y = np.concatenate(y, axis=0)

            print(key, X.shape)

            ds = tf.data.Dataset.from_tensor_slices((X, y))
            tf.data.experimental.save(dataset=ds, path=str(os.path.join(cache_folder, key + '.tfds')))
            with open(str(os.path.join(cache_folder, key + '.pkl')), 'wb') as handle:
                pickle.dump(ds.element_spec, handle)

    with open(str(os.path.join(cache_folder, 'train.pkl')), 'rb') as handle:
         train_element_spec = pickle.load(handle)
    train_ds = tf.data.experimental.load(path=str(os.path.join(cache_folder, 'train.tfds')), element_spec=train_element_spec)
    with open(str(os.path.join(cache_folder, 'val.pkl')), 'rb') as handle:
         val_element_spec = pickle.load(handle)
    val_ds = tf.data.experimental.load(path=str(os.path.join(cache_folder, 'val.tfds')), element_spec=val_element_spec)
    with open(str(os.path.join(cache_folder, 'test.pkl')), 'rb') as handle:
         test_element_spec = pickle.load(handle)
    test_ds = tf.data.experimental.load(path=str(os.path.join(cache_folder, 'test.tfds')), element_spec=test_element_spec)

    return train_ds, val_ds, test_ds









if __name__ == '__main__':
    raw_data = data(dataset='final_data_2', recache=False)
    raw_data.train_val_test(multi_comp=True, val_comp_size=0.1, test_comp_size=0.07, val_time_size=0.2, test_time_size=0.1)
    train_ds, val_ds, test_ds = make_dataset(input_width=20, label_width=4, shift=1, label_columns=[raw_data.props['y_col']], train_df=raw_data.train_sorted, val_df=raw_data.val_sorted, test_df=raw_data.test_sorted, data_hash=raw_data.data_hash)

    print(train_ds, len(train_ds))
