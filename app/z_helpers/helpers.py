import sys, os, pathlib, warnings, datetime
import datetime as dt
import tables as tb
import pandas as pd
import numpy as np
import psycopg2, psycopg2.extras
import configparser
from silx.io import dictdump


######################################### GENERAL #########################################


def change_workdir():
    new_wkdir = os.path.dirname(os.path.dirname(__file__))
    os.chdir(new_wkdir)
    print('Working dir changed to:', os.getcwd())



def convenience_settings():
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width', 300)
    print('Custom parameters set')
    change_workdir()


def get_project_directories(key=None, **kwargs):
    helpers_path = os.path.dirname(__file__)
    app_path = os.path.dirname(helpers_path)
    project_path = os.path.dirname(app_path)
    data_path = os.path.join(project_path, 'data')
    cache_path = os.path.join(project_path, 'cache')
    credentials_path = os.path.join(project_path, 'credentials')
    working_dir = os.getcwd()
    tensorboard_logs = '/Users/vanalmsick/opt/anaconda3/envs/dev/lib/python3.8/site-packages/tensorboard/logs'

    dir_dict = {'project_dir': project_path, 'app_dir': app_path, 'data_dir': data_path, 'cred_dir':credentials_path, 'helpers_dir': helpers_path, 'working_dir': working_dir, 'tensorboard_logs':tensorboard_logs, 'cache_dir':cache_path}

    if key is None:
        return dir_dict
    else:
        return dir_dict[key]


def get_curr_time():
    return dt.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

def get_start_time():
    return get_curr_time()


def get_credentials(credential=None, cred_dir=None, dir_dict=None, **kwargs):
    if cred_dir is None and dir_dict is None:
        cred_dir = get_project_directories(key='cred_dir')
    elif cred_dir is None and dir_dict is not None:
        cred_dir = dir_dict['cred_dir']

    file_list = [f for f in os.listdir(cred_dir) if os.path.isfile(os.path.join(cred_dir, f))]
    creds = {}

    for file in file_list:
        if file[-4:] == '.ini':
            config = configparser.ConfigParser()
            config.read(os.path.join(cred_dir, file))
            conf_dict = config._sections
            if len(conf_dict.keys()) == 1:
                key = list(conf_dict.keys())[0]
                creds[key] = conf_dict[key]
            else:
                creds[file.split('.')[0]] = conf_dict
        elif file[-4:] == '.key':
            with open(os.path.join(cred_dir, file), 'r') as f:
                creds[file.split('.')[0]] = f.read()

    if credential is None:
        return creds
    else:
        return creds[credential]



######################################### SQL/Database #########################################


def postgresql_connect(params_dic):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params_dic)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1)
    print("Connection successful")
    return conn


def _dtype_mapping():
    return {'object' : 'TEXT',
        'string': 'TEXT',
        'int64' : 'REAL', # INT
        'float64' : 'REAL',
        'datetime64' : 'DATETIME',
        'bool' : 'BOOLEAN',
        'category' : 'TEXT',
        'timedelta[ns]' : 'TIMESTAMP'}



def _gen_tbl_cols_sql(df):
    dmap = _dtype_mapping()
    sql = ""
    df1 = df.rename(columns = {"" : "nocolname"})
    hdrs = df1.dtypes.index
    hdrs_list = [(hdr, str(df1[hdr].dtype)) for hdr in hdrs]
    for i, hl in enumerate(hdrs_list):
        sql += " ,{0} {1}".format(hl[0].lower(), dmap[str(hl[1]).lower()])
    return sql[2:]


def create_sql_tbl(df, conn, tbl_name, schema='public'):
    if check_table_exists(conn=conn, table_name=tbl_name, schema=schema):
        print('Table {}.{} already exists.'.format(schema, tbl_name))
    else:
        tbl_cols_sql = _gen_tbl_cols_sql(df)
        sql = "CREATE TABLE {}.{} ({})".format(schema, tbl_name, tbl_cols_sql)
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        conn.commit()


def check_table_exists(conn, table_name, schema='public'):
    sql = """SELECT EXISTS (
               SELECT FROM pg_catalog.pg_class c
               JOIN   pg_catalog.pg_namespace n ON n.oid = c.relnamespace
               WHERE  n.nspname = '{schema}'
               AND    c.relname = '{table}'
               AND    c.relkind = 'r'    -- only tables
               )""".format(schema=schema, table=table_name)
    with conn.cursor() as cur:
        cur.execute(sql)
        answer = cur.fetchall()[0][0]
    return answer


def sql_query(sql, conn):
    return pd.read_sql_query(sql, conn)


def df_insert_sql(conn, df, table, schema='public'):
    """
    Using psycopg2.extras.execute_values() to insert the dataframe
    """
    # Create a list of tupples from the dataframe values
    df = df.replace({pd.NaT: np.nan, '':np.nan})
    for col in (df.select_dtypes(include=[np.datetime64])).columns.tolist():
        df[col] = [d if not pd.isnull(d) else None for d in df[col]]
    warnings.filterwarnings('ignore')
    np_array = df.replace({pd.np.nan: None}).values
    warnings.filterwarnings('default')
    tuples = [tuple(x) for x in np_array]
    # Comma-separated dataframe columns
    cols = ','.join(df.columns.tolist())
    # SQL quert to execute
    query  = "INSERT INTO %s.%s (%s) VALUES %%s" % (schema, table, cols)
    cursor = conn.cursor()
    try:
        psycopg2.extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1, error
    #print("execute_values() done")
    cursor.close()


def update_request_status(request_id, conn, schema, new_status=1):
    if type(request_id) != list:
        request_id = [request_id]

    with conn.cursor() as cur:
        for id in request_id:
            updated = datetime.datetime.now().isoformat()
            SQL = """UPDATE {schema}.data_request_list SET status = {status}, last_updated = '{updated}' WHERE request_id = {id}""".format(status=new_status, schema=schema, updated=updated, id=id)
            cur.execute(SQL)
    conn.commit()

def submit_error(error_dict, conn, schema):
    request_id = error_dict['request_id']
    error_type = error_dict['error_type']
    error_message = str(error_dict['error_message']).replace("'",'"')

    if 'timestamp_iso' in error_dict:
        error_time = error_dict['timestamp_iso']
    else:
        error_time = datetime.datetime.now().isoformat()

    with conn.cursor() as cur:
        SQL = """INSERT INTO {schema}.data_request_errors (request_id, error_time, error_type, error_message) VALUES ({request_id}, '{error_time}', '{error_type}', '{error_message}')""".format(schema=schema, request_id=request_id, error_time=error_time, error_type=error_type, error_message=error_message)
        cur.execute(SQL)
    conn.commit()


def create_wrds_sql_query(library='comp', table='company', columns='*', conditions=None, custom_condition='', distinct=False, no_observations=-1, offset=0):
    """Libary options: libary=['fundq', 'funda', 'company']"""
    if distinct:
        dis = ' DISTINCT '
    else:
        dis = ''
    if no_observations < 0:
        obsstmt = ''
    else:
        obsstmt = 'LIMIT {}'.format(no_observations)
    if columns is None:
        cols = '*'
    else:
        cols = ','.join(columns)
    if conditions is None:
        cond = ' '
    else:
        cond = 'WHERE'
        if custom_condition != '':
            cond = cond + ' ' + custom_condition + ' AND'
        for key in conditions:
            if type(conditions[key]) == list:
                cond = cond + ' ' + key + ' in (' + str(conditions[key])[1:-1] + ') AND'
            elif type(conditions[key]) == str:
                cond = cond + ' ' + key + " = '" + str(conditions[key]) + "' AND"
            elif type(conditions[key]) == int or type(conditions[key]) == float:
                cond = cond + ' ' + key + " = " + str(conditions[key]) + " AND"
            else:
                raise Exception('Unknown data type of condition.')
        cond = cond[:-3]

    sql = ("SELECT{distinct} {cols} FROM {schema}.{table} {condition} {obsstmt} OFFSET {offset};".format(cols=cols, distinct=dis, schema=library, table=table, condition=cond, obsstmt=obsstmt, offset=offset))
    return sql



class custom_hdf5:
    def dict_to_hdf5(file, save_dict):
        create_ds_args = {'compression': "gzip", 'fletcher32': True}
        dictdump.dicttoh5(save_dict, file, h5path="/", mode='w', create_dataset_args=create_ds_args)
        #dictdump.dicttoh5(save_dict, file, h5path="/", mode='w')


    def hdf5_to_dict(file, *args):
        hdf_path = '/' + '/'.join(args[:-1])
        out = dictdump.h5todict(file, hdf_path)
        out = out[args[-1]]
        return out

    def hdf5_to_pd(file, *args):
        my_dict = custom_hdf5.hdf5_to_dict(file, *args)
        pd_ser = pd.Series(dict(zip(my_dict[0, :].astype(str), my_dict[1, :].astype(float))))
        return pd_ser

    def pd_series_to_2d_array(pd_series):
        np_array = np.array([pd_series.index.values.astype(str), pd_series.values.astype(float)])
        return np_array

    def get_comp_list(file, norm_key):
        import h5py
        with h5py.File(file, 'r') as f:
            a = list(f.get(norm_key).keys())
        try:
            a.remove('__all__')
        except:
            pass
        return a




if __name__ == '__main__':
    #print(get_project_directories())
    #print(get_project_directories('project_dir'))
    #print(get_credentials())
    #print(get_credentials('reuters_eikon_api'))

    get_credentials()



