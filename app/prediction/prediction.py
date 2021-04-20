import os, sys
import psycopg2, psycopg2.extras
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime as dt
import shutil, time, math
import matplotlib.pyplot as plt


### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import helpers as my
##################################





def download_data_from_sql(data_version='default', recache=False):
    query = "SELECT * FROM selected_data WHERE report_type = 'FQ'"
    cache_folder = "cache/" + str(data_version) + '/'

    param_dic = my.get_credentials(credential='local_databases')['reuters']

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)


    if recache or not os.path.exists((cache_folder + 'raw_data.csv')):
        print('Getting raw data via sql...')

        with my.postgresql_connect(param_dic) as conn:
            df = pd.read_sql_query(query, con=conn)
            df.drop(columns=['tr_revenue_date', 'tr_bsperiodenddate', 'report_type', 'request_id', 'last_updated'], inplace=True)
            df.to_csv((cache_folder + 'raw_data.csv'), index=False)
        print('Raw data cached.')

    else:
        print('Raw data already cached.')


def df_to_dataset(dataframe, target_col='tr_f_ebit', shuffle=True, batch_size=32):
    dataframe = dataframe.copy()

    # Use Pandas dataframe's pop method to get the list of targets.
    labels = dataframe.pop(target_col)

    # Create a tf.data.Dataset from the dataframe and labels.
    ds = tf.data.Dataset.from_tensor_slices(((dict(dataframe), labels.values)))

    if shuffle:
        # Shuffle dataset.
        ds = ds.shuffle(len(dataframe))

    # Batch dataset with specified batch_size parameter.
    ds = ds.batch(batch_size)

    return ds



class data_iterator:
    def __init__(self, df, iter_col='index', history_size=4*4, target_size=4*2, steps=1, shuffle=True):
        self.data = df.copy()
        self.history_size = history_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.i = 0
        if iter_col != 'index':
            self.data.set_index(iter_col, inplace=True)
        self.data.sort_index(inplace=True)
        self.iter_list = list(np.unique(self.data.index.values))
        self.lower_idx = list(range(0, len(self.iter_list) - history_size - target_size + 1, steps))
        self.upper_idx = list(range(history_size + target_size - 1, len(self.iter_list), steps))

    def __iter__(self):
        return self

    def __next__(self):
        if self.i <= len(self.lower_idx) - 1:
            current_df = self.data[(self.data.index >= self.iter_list[self.lower_idx[self.i]]) & (self.data.index <= self.iter_list[self.upper_idx[self.i]])]
            self.i += 1
            return current_df
        else:
            raise StopIteration


def normalize_data(df, exclude_cols=[]):
    # ToDo: Ln Normalization??? Outliers?
    df = df.copy()
    normalize_col_list = [i for i in df.columns.tolist() if pd.api.types.is_numeric_dtype(df[i]) and i not in exclude_cols]
    data_mean = df[normalize_col_list].mean()
    data_std = df[normalize_col_list].std()
    df[normalize_col_list] = (df[normalize_col_list] - data_mean) / data_std
    normalization_param = {'mean': data_mean, 'std': data_std}
    return df, normalization_param


def split_data(df, batch_sep_col='stock', history_size=4*2, target_size=4*2, y_col='tr_f_ebit', drop_cols=[]):
    X = df.copy()
    X.drop(columns=drop_cols, inplace=True)
    X_input = []
    y_output = []
    y_hist = []
    X_cols = X.columns.tolist()
    X_cols.remove(batch_sep_col)
    inxed_list = list(np.unique(X[batch_sep_col]))
    for idx in inxed_list:
        X_tmp = X[X[batch_sep_col] == idx][X_cols]
        if X_tmp.shape[0] == (history_size + target_size):
            X_input.append(np.array(X_tmp)[:-target_size, :])
            y_tmp = X[X[batch_sep_col] == idx][y_col]
            y_output.append(np.atleast_2d(np.array(y_tmp))[:, history_size:])
            y_hist.append(np.atleast_2d(np.array(y_tmp))[:, :history_size])
    X_input = tf.stack(X_input)
    y_output = tf.stack(y_output)
    y_hist = tf.stack(y_hist)
    return X_input, y_output, y_hist





def compile_and_fit(model, X, y, patience=100, model_name='UNKNOWN', MAX_EPOCHS=20):
  tracking_address = my.get_project_directories(key='tensorboard_logs')
  TBLOGDIR = tracking_address + "/" + model_name

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TBLOGDIR, histogram_freq=1)
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                    patience=patience,
                                                    mode='min',
                                                    restore_best_weights=True)
  model.compile(loss=tf.losses.MeanAbsolutePercentageError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanAbsolutePercentageError()])

  history = model.fit(X, y, epochs=MAX_EPOCHS,callbacks=[early_stopping, tensorboard_callback])

  return history


def plot(model, y_true, y_hist, normalization_param, y_col, example=0):
    scale = 1000000

    if type(example) != list:
        example = [example]

    plt.figure(figsize=(12, 3 * len(example)))

    for i in range(len(example)):
        ex = example[i]

        plt.subplot(len(example), 1, i + 1)
        plt.ylabel(f'{y_col} [example {ex}]')

        y_pred_i = model.predict(X)[ex, -1, :]
        y_true_i = y_true[ex, :, :][0]
        y_hist_i = np.array(y_hist[ex, :, :])[0]
        mean = normalization_param['mean'][y_col]
        std = normalization_param['std'][y_col]
        y_pred_real = y_pred_i * std + mean
        y_true_real = y_true_i * std + mean
        y_hist_real = y_hist_i * std + mean
        x_hist = [int(i) for i in list(range(0, len(y_hist_real)))]
        x_pred = [int(i) for i in list(range(len(y_hist_real), len(y_hist_real) + len(y_true_real)))]
        plt.plot(x_hist, (y_hist_real / scale), label='Historical', marker='.', zorder=-10)
        plt.scatter(x_pred, (y_true_real / scale), edgecolors='k', label='True', c='#2ca02c', s=64)
        plt.scatter(x_pred, (y_pred_real / scale), marker='X', edgecolors='k', label='Predictions (in-sample/test)', c='#ff7f0e', s=64)

        if i == 0:
            plt.legend()

        xint = range(min(x_hist), math.ceil(max(x_pred)) + 1)
        plt.xticks(xint)

    plt.show()


def feature_engineering(path):
    raw_data = pd.read_csv(path)
    raw_data.fillna(0, inplace=True)

    raw_data['period_date'] = pd.to_datetime(raw_data['period_date'], format='%Y-%m-%d')

    raw_data['qrt_sin'] = raw_data['period_quarter'].map({1.0:1, 2.0:0, 3.0:-1, 4.0:0})
    raw_data['qrt_cos'] = raw_data['period_quarter'].map({1.0: 0, 2.0: -1, 3.0: 0, 4.0: 1})

    return raw_data


if __name__ == '__main__':
    my.convenience_settings()
    data_version = 'default'
    recache = False
    iter_col = ['period_year', 'period_quarter']
    history_size = 4*4
    target_size = 4
    y_col = 'tr_f_ebit'
    drop_cols = ['period_date']
    batch_sep_col = 'stock'

    ### run tensorboard
    tracking_address = my.get_project_directories(key='tensorboard_logs')
    try:
        shutil.rmtree(tracking_address)
        time.sleep(10)
    except:
        pass
    os.mkdir(tracking_address)


    cache_folder = "prediction/cache/" + str(data_version) + '/'
    download_data_from_sql(data_version=data_version, recache=recache)

    raw_data = feature_engineering(path=(cache_folder + 'raw_data.csv'))

    data = data_iterator(df=raw_data, iter_col=iter_col, history_size=history_size, target_size=target_size)
    for partial_data in data:
        min_date = partial_data['period_date'].min().strftime('%y-%m')
        max_date = partial_data['period_date'].max().strftime('%y-%m')
        period = min_date + '_' + max_date
        print(period)

        normalized_data, normalization_param = normalize_data(df=partial_data, exclude_cols=['period_date'])
        X, y, y_hist = split_data(df=normalized_data, batch_sep_col=batch_sep_col, history_size=history_size, target_size=target_size, y_col=y_col, drop_cols=drop_cols)

        linear = tf.keras.Sequential([
                                    tf.keras.layers.LSTM(117, return_sequences=True),
                                    tf.keras.layers.Dense(target_size * 1, kernel_initializer=tf.initializers.zeros()),
                                    tf.keras.layers.Dense(target_size)
                                    ])

        history = compile_and_fit(X=X, y=y, model=linear, MAX_EPOCHS=5000, model_name=('LSTM:'+period))

        plot(model=linear, y_true=y, y_hist=y_hist, normalization_param=normalization_param, y_col=y_col, example=[0, 5, 6, 20, 100])
        #raise Excption('fghj')

