import os, sys
import pandas as pd
import tensorflow as tf
import shutil, time
import sklearn.decomposition

import prediction as my_pred

### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import helpers as my
##################################


class prepare_data:
    def __init__(self, y_col, pred_len, exclude_cols=[], append_cols=[], pca=0.99):
        self.exclude_cols = exclude_cols + append_cols
        self.append_cols = append_cols + [y_col]
        self.y_col = y_col
        self.pred_len = pred_len
        self.pca = pca

    def fit(self, train_df):
        train_df = train_df.fillna(0)
        self.normalize_col_list = [i for i in train_df.columns.tolist() if pd.api.types.is_numeric_dtype(train_df[i]) and i not in self.exclude_cols]
        self.data_mean = train_df[self.normalize_col_list].mean()
        self.data_std = train_df[self.normalize_col_list].std()
        df_new = (train_df[self.normalize_col_list] - self.data_mean) / self.data_std
        df_new = df_new.fillna(0)
        if self.pca != False:
            self.pca_obj = sklearn.decomposition.PCA(self.pca)
            df_new = pd.DataFrame(self.pca_obj.fit_transform(df_new), index=train_df.index)
        df_new = pd.concat([df_new, train_df[self.append_cols]], axis=1)
        X = df_new.iloc[:-self.pred_len].loc[:, df_new.columns != self.y_col].values
        y = df_new.iloc[-self.pred_len:][self.y_col].values
        X = tf.stack([X])
        y = tf.stack([y])
        return X, y

    def transform(self, df):
        df = df.fillna(0)
        df_new = (df[self.normalize_col_list] - self.data_mean) / self.data_std
        df_new = df_new.fillna(0)
        if self.pca != False:
            df_new = pd.DataFrame(self.pca_obj.transform(df_new), index=df.index)
        df_new = pd.concat([df_new, df[self.append_cols]], axis=1)
        X = df_new.iloc[:-self.pred_len].loc[:, df_new.columns != self.y_col].values
        y = df_new.iloc[-self.pred_len:][self.y_col].values
        X = tf.stack([X])
        y = tf.stack([y])
        return X, y






if __name__ == '__main__':
    my.convenience_settings()
    data_version = 'default'
    recache = False
    iter_col = ['period_year', 'period_qrt']
    history_size = 4*4
    target_size = 4
    y_col = 'tr_f_ebit'
    category_cols = ['gsector']
    drop_cols = ['ric', 'curcdq', 'rp', 'updq', 'iid', 'exchg', 'costat', 'fic', 'srcq', 'curncdq', 'acctsdq', 'acctstdq', 'ggroup'] + category_cols
    batch_sep_col = 'gvkey'

    ### run tensorboard
    tracking_address = my.get_project_directories(key='tensorboard_logs')
    try:
        shutil.rmtree(tracking_address)
        time.sleep(10)
    except:
        pass
    os.mkdir(tracking_address)


    cache_folder = "cache/" + str(data_version) + '/'
    my_pred.download_data_from_sql(data_version=data_version, recache=recache)

    raw_data = my_pred.feature_engineering(path=(cache_folder + 'raw_data.csv'), category_cols=category_cols)
    raw_data.set_index([batch_sep_col] + iter_col, inplace=True)
    raw_data.sort_index(inplace=True)

    data_types = raw_data.columns.to_series().groupby(raw_data.dtypes).groups
    print('Columns dtypes:', data_types)

    back_looking = 4*5
    prediction = 4
    length = back_looking + prediction * 3

    for company, df in raw_data.groupby(level=0):
        max_idx = len(df)
        for i, j in zip(range(0, max_idx - length + 1, 1), range(length, max_idx + 1, 1)):
            preparer = prepare_data(y_col=y_col, pred_len=4, exclude_cols=['ric','ggroup'], append_cols=['qrt_sin','qrt_cos'], pca=0.99)
            X_train, y_train = preparer.fit(df.iloc[i:j-prediction*2])
            X_val, y_val = preparer.transform(df.iloc[i+prediction:j-prediction])
            X_test, y_test = preparer.transform(df.iloc[i+prediction*2:j])

            input_layer_shape = X_train.shape[1]
            linear = tf.keras.Sequential([
                # tf.keras.layers.Dense(int(input_layer_shape*0.75)),
                # tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.LSTM(input_layer_shape, return_sequences=True),
                # tf.keras.layers.LSTM(int(input_layer_shape * 0.75), return_sequences=False),
                # tf.keras.layers.Dropout(rate=0.2),
                # tf.keras.layers.LSTM(input_layer_shape, return_sequences=True),
                # tf.keras.layers.Dense(target_size * 1, kernel_initializer=tf.initializers.ones()),
                tf.keras.layers.Dense(target_size * 1, kernel_initializer=tf.initializers.zeros())
            ])

            history = my_pred.compile_and_fit(X_train=X_train, y_train=y_train,
                                      X_val=X_val, y_val=y_val, model=linear,
                                      MAX_EPOCHS=5000, model_name=('LSTM_' + str(company) + str(i) + '-' + str(j)))

            print("Evaluate on test data")
            results = linear.evaluate(X_test, y_test)
            print("test loss, test acc:", results)

