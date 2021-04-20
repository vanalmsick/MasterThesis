import sys, os, pathlib
import datetime as dt
import tables as tb
import pandas as pd
import psycopg2, psycopg2.extras
import configparser


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
    helpers_path = os.path.abspath(os.getcwd())
    app_path = os.path.dirname(helpers_path)
    project_path = os.path.dirname(app_path)
    data_path = os.path.join(project_path, 'data')
    credentials_path = os.path.join(project_path, 'credentials')
    working_dir = os.getcwd()
    tensorboard_logs = '/Users/vanalmsick/opt/anaconda3/envs/dev/lib/python3.8/site-packages/tensorboard/logs'

    dir_dict = {'project_dir': project_path, 'app_dir': app_path, 'data_dir': data_path, 'cred_dir':credentials_path, 'helpers_dir': helpers_path, 'working_dir': working_dir, 'tensorboard_logs':tensorboard_logs}

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
        'int64' : 'INT',
        'float64' : 'FLOAT',
        'datetime64' : 'DATETIME',
        'bool' : 'TINYINT',
        'category' : 'TEXT',
        'timedelta[ns]' : 'TEXT'}



def _gen_tbl_cols_sql(df):
    dmap = _dtype_mapping()
    sql = "pi_db_uid SERIAL"
    df1 = df.rename(columns = {"" : "nocolname"})
    hdrs = df1.dtypes.index
    hdrs_list = [(hdr, str(df1[hdr].dtype)) for hdr in hdrs]
    for i, hl in enumerate(hdrs_list):
        sql += " ,{0} {1}".format(hl[0], dmap[hl[1]])
    return sql


def create_sql_tbl(df, conn, tbl_name):
    tbl_cols_sql = _gen_tbl_cols_sql(df)
    sql = "CREATE TABLE {0} ({1})".format(tbl_name, tbl_cols_sql)
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()
    conn.commit()







if __name__ == '__main__':
    #print(get_project_directories())
    #print(get_project_directories('project_dir'))
    #print(get_credentials())
    #print(get_credentials('reuters_eikon_api'))

    get_credentials()



