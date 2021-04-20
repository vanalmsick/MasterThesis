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
    def __init__(self, df, iter_col='index', history_size=4*4, target_size=4*2, steps=5, shuffle=True):
        self.data = df.copy()
        self.history_size = history_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.i = 0
        if iter_col != 'index':
            self.data.set_index(iter_col, inplace=True)
        self.data.sort_index(inplace=True)
        self.iter_list = list(np.unique(self.data.index.values))
        self.lower_idx = list(range(0, len(self.iter_list) - history_size - (target_size * 2) + 1, steps))
        self.upper_idx = list(range(history_size + (target_size * 2) - 1, len(self.iter_list), steps))

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


def split_data(df, batch_sep_col='stock', history_size=4*2, target_size=4*2, val_spit=0.2, y_col='tr_f_ebit', drop_cols=[], shuffle=True):
    copy_df = df.copy()
    copy_df[batch_sep_col] = copy_df[batch_sep_col].astype("category")
    drop_cols = drop_cols + [batch_sep_col]

    # Seperate Train and Validation data randomly
    batch_col_list = np.array(copy_df[batch_sep_col].unique())
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(batch_col_list)
    split_point = int(val_spit * len(batch_col_list))
    train_batch = batch_col_list[:-split_point]
    val_batch = batch_col_list[-split_point:]
    test_batch = batch_col_list

    date_list = np.unique(copy_df.index.values)
    date_list.sort()
    if len(date_list) != history_size + target_size * 2:
        raise Exception('Unexpected date length of data.')
    train_val_dates = date_list[:history_size + target_size]
    test_dates = date_list[-(history_size + target_size):]

    train_val_data = copy_df.loc[train_val_dates]
    test_data = copy_df.loc[test_dates]

    X_cols = [i for i in copy_df.columns.tolist() if i not in drop_cols]

    # Training data
    X_train = []
    y_train_hist = []
    y_train = []
    for idx in train_batch:
        X_tmp = train_val_data[train_val_data[batch_sep_col] == idx][X_cols]
        if X_tmp.shape[0] == (history_size + target_size):
            y_tmp = train_val_data[train_val_data[batch_sep_col] == idx][y_col]
            y_train.append(np.atleast_2d(np.array(y_tmp))[:, history_size:])
            y_train_hist.append(np.atleast_2d(np.array(y_tmp))[:, :history_size])
            X_train.append(np.array(X_tmp)[:-target_size, :])
    X_train = tf.stack(X_train)
    y_train_hist = tf.stack(y_train_hist)
    y_train = tf.stack(y_train)


    # Validation data
    X_val = []
    y_val_hist = []
    y_val = []
    for idx in val_batch:
        X_tmp = train_val_data[train_val_data[batch_sep_col] == idx][X_cols]
        if X_tmp.shape[0] == (history_size + target_size):
            y_tmp = train_val_data[train_val_data[batch_sep_col] == idx][y_col]
            y_val.append(np.atleast_2d(np.array(y_tmp))[:, history_size:])
            y_val_hist.append(np.atleast_2d(np.array(y_tmp))[:, :history_size])
            X_val.append(np.array(X_tmp)[:-target_size, :])
    X_val = tf.stack(X_val)
    y_val_hist = tf.stack(y_val_hist)
    y_val = tf.stack(y_val)


    # test data
    X_test = []
    y_test_hist = []
    y_test = []
    for idx in test_batch:
        X_tmp = test_data[test_data[batch_sep_col] == idx][X_cols]
        if X_tmp.shape[0] == (history_size + target_size):
            y_tmp = test_data[test_data[batch_sep_col] == idx][y_col]
            y_test.append(np.atleast_2d(np.array(y_tmp))[:, history_size:])
            y_test_hist.append(np.atleast_2d(np.array(y_tmp))[:, :history_size])
            X_test.append(np.array(X_tmp)[:-target_size, :])
    X_test = tf.stack(X_test)
    y_test_hist = tf.stack(y_test_hist)
    y_test = tf.stack(y_test)


    out_data = {'train': {'X':X_train, 'y':y_train, 'y_hist':y_train_hist},
                'val': {'X':X_val, 'y':y_val, 'y_hist':y_val_hist},
                'test': {'X':X_test, 'y':y_test, 'y_hist':y_test_hist}}


    return out_data





def compile_and_fit(model, X_train, y_train, X_val=None, y_val=None, patience=250, model_name='UNKNOWN', MAX_EPOCHS=20):
    tracking_address = my.get_project_directories(key='tensorboard_logs')
    TBLOGDIR = tracking_address + "/" + model_name

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TBLOGDIR, histogram_freq=1)
    model.compile(loss=tf.losses.MeanAbsoluteError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanAbsolutePercentageError()])

    if X_val is not None and y_val is not None:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min',
                                                          restore_best_weights=True)
        training_history = model.fit(X_train, y_train, epochs=MAX_EPOCHS, validation_data=(X_val, y_val), callbacks=[early_stopping, tensorboard_callback])
    else:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          patience=patience,
                                                          mode='min',
                                                          restore_best_weights=True)
        training_history = model.fit(X_train, y_train, epochs=MAX_EPOCHS, callbacks=[early_stopping, tensorboard_callback])

    return training_history


def plot(y_hist, y_pred, y_true, normalization_param, y_name='tr_f_ebit', model_name='Model', examples=0):
    scale = 1000000

    if type(examples) != list:
        examples = [examples]

    plt.figure(figsize=(12, 3 * len(examples)))

    for i in range(len(examples)):
        ex = examples[i]

        plt.subplot(len(examples), 1, i + 1)
        plt.ylabel(f'{y_name} [example {ex}]')

        mean = normalization_param['mean'][y_name]
        std = normalization_param['std'][y_name]
        y_pred_real = np.array(y_pred[i, -1, :]) * std + mean
        y_true_real = np.array(y_true[i, -1, :]) * std + mean
        y_hist_real = np.array(y_hist[i, -1, :]) * std + mean
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
        data_splitted = split_data(df=normalized_data, batch_sep_col=batch_sep_col, history_size=history_size, target_size=target_size, y_col=y_col, drop_cols=drop_cols)

        linear = tf.keras.Sequential([
                                    tf.keras.layers.LSTM(117, return_sequences=True),
                                    tf.keras.layers.Dense(target_size * 1, kernel_initializer=tf.initializers.zeros()),
                                    tf.keras.layers.Dense(target_size)
                                    ])

        history = compile_and_fit(X_train=data_splitted['train']['X'], y_train=data_splitted['train']['y'], X_val=data_splitted['val']['X'], y_val=data_splitted['val']['y'], model=linear, MAX_EPOCHS=5000, model_name=('LSTM_' + period))

        print("Evaluate on test data")
        results = linear.evaluate(data_splitted['test']['X'], data_splitted['test']['y'])
        print("test loss, test acc:", results)

        print("Generate predictions for 3 samples")
        example_comps = [0, 5, 6, 20, 100]
        predictions = linear.predict(tf.gather(data_splitted['test']['X'], example_comps))
        print("predictions shape:", predictions.shape)

        plot(y_hist=tf.gather(data_splitted['test']['y_hist'], example_comps), y_pred=predictions, y_true=tf.gather(data_splitted['test']['y'], example_comps), y_name=y_col, normalization_param=normalization_param, examples=example_comps)
        raise Excption('fghj')

