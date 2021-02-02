import sys, os
import datetime as dt
import tables as tb
import pandas as pd

def get_project_directories(key=None, **kwargs):
    helpers_path = os.path.abspath(os.getcwd())
    app_path = os.path.dirname(helpers_path)
    project_path = os.path.dirname(app_path)
    data_path = os.path.join(project_path, 'data')
    credentials_path = os.path.join(project_path, 'credentials')

    dir_dict = {'project_dir': project_path, 'app_dir': app_path, 'data_dir': data_path, 'cred_dir':credentials_path, 'helpers_dir': helpers_path}

    if key is None:
        return dir_dict
    else:
        return dir_dict[key]



def get_credentials(cred=None, cred_dir=None, dir_dict=None, **kwargs):
    if cred_dir is None and dir_dict is None:
        cred_dir = get_project_directories(key='cred_dir')
    elif cred_dir is None and dir_dict is not None:
        cred_dir = dir_dict['cred_dir']

    file_list = [f for f in os.listdir(cred_dir) if os.path.isfile(os.path.join(cred_dir, f))]
    creds = {}

    for file in file_list:
        with open(os.path.join(cred_dir, file), 'r') as f:
            creds[file.split('.')[0]] = f.read()

    if cred is None:
        return creds
    else:
        return creds[cred]




class database():
    def __init__(self, database, data_dir=None):
        if data_dir is None:
            data_dir = get_project_directories('data_dir')
        self.data_dir = data_dir
        self.hdf5_dir = os.path.join(self.data_dir, (database + '.h5'))

        self.hdf = pd.HDFStore(self.hdf5_dir, mode='a')


    def append(self, df):
        needed_cols = {'ref_date':'dateint', 'stock_symbol':'category', 'report_type':'category', 'data_label':object, 'data_value':float}

        if 'last_updated' not in df.columns.to_list():
            df['last_updated'] = int(dt.datetime.now().strftime('%Y%m%d%H%M%S'))

        for col in needed_cols.keys():
            if not col in df.columns.to_list():
                raise Exception('Column {} is missing in df.'.format(col))

            if needed_cols[col] == 'dateint':
                if df.dtypes[col] != int or not (df[col].min() >= 19000101000000 and df[col].max() < 20220101000000):
                    raise Exception('Column {} has not data-type dateint which is "%Y%m%d%H%M%S".'.format(col))
            elif needed_cols[col] == 'category':
                if str(df.dtypes[col]) != 'category':
                    raise Exception(
                        'Column {} has data-type {} but should have category please correct.'.format(col, df.dtypes[col]))
            else:
                if df.dtypes[col] != needed_cols[col]:
                    raise Exception('Column {} has data-type {} but should have {} please correct.'.format(col, df.dtypes[col], needed_cols[col]))

        df.set_index(['ref_date', 'stock_symbol', 'report_type', 'data_label'], inplace=True)
        self.hdf.append('/', df, format='table')


    def read(self):
        return self.hdf.get('/')

    def get(self):
        return self.read()

    def select(self, **kwargs):
        return self.hdf.select(key='/', **kwargs)

    def info(self):
        return self.hdf.info()

    def keys(self):
        return self.hdf.keys()

    def __exit__(self):
        self.hdf.close()





if __name__ == '__main__':
    print(get_project_directories())
    print(get_project_directories('project_dir'))
    print(get_credentials())
    print(get_credentials('reuters_eikon_api'))

    data = database(database='test')
    df = pd.DataFrame({'ref_date':[20200202162101,20200202162101,20200202162101,20200202162101], 'stock_symbol':['APPL','AAPL','AAPL','APPL'], 'report_type':['A','Q','A','Q'], 'data_label':['fdgdf','fdgdf','fdgdf','fdgdf'], 'data_value':[0.,1.,2.,3.]})
    df['stock_symbol'] = df['stock_symbol'].astype("category")
    df['report_type'] = df['report_type'].astype("category")
    print(data.append(df))
    print(data.get())
    #print(data.select(where='index > 2'))
