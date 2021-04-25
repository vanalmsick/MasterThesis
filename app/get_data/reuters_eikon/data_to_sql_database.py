import datetime

import pandas as pd
import numpy as np
import pickle
import os, json, psycopg2, psycopg2.extras, sys, math, warnings


### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import helpers as my
##################################





def df_insert_sql(conn, df, table):
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
    query  = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
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


def df_rename(df, json_data):
    head = json_data['headers'][0]
    col_list = df.columns.tolist()
    col_should_be = [i['displayName'] for i in head]
    if col_should_be[0] is None:
        col_should_be[0] = 'stock'
    added_cols = ['report_type', 'request_id', 'last_updated']
    col_should_be = col_should_be + added_cols
    new_col_list = ['stock'] + [i['field'].replace('.','_') for i in head[1:]] + added_cols
    if (col_list == col_should_be) == False:
        # create df from json because some error in df
        col_names = ['stock'] + [i['field'].replace('.','_') for i in head[1:]]
        df = pd.DataFrame.from_records(json_data['data'], columns=col_names)
        all_is_none = df[col_names[1:]].isnull().values.all()
        if all_is_none:
            # Dataframe exists just of None s
            df = None
    else:
        # Replacing column names by position
        df.columns = new_col_list
    return df


def update_request_status(request_id, conn, new_status=1):
    if type(request_id) != list:
        request_id = [request_id]

    with conn.cursor() as cur:
        for id in request_id:
            SQL = """UPDATE request_list SET status = {status} WHERE request_id = {id}""".format(status=new_status, id=id)
            cur.execute(SQL)
    conn.commit()


def submit_error(error_dict, conn):
    request_id = error_dict['task_id']
    error_time = error_dict['timestamp_iso']
    error_type = error_dict['error_type']
    error_message = str(error_dict['error_code']).replace("'",'"')
    with conn.cursor() as cur:
        SQL = """INSERT INTO request_errors (request_id, error_time, error_type, error_message) VALUES ({request_id}, '{error_time}', '{error_type}', '{error_message}')""".format(request_id=request_id, error_time=error_time, error_type=error_type, error_message=error_message)
        cur.execute(SQL)
    conn.commit()


def data_to_sql(folder_path, offset_request_id=0):
    files = os.listdir(folder_path)

    param_dic = my.get_credentials(credential='local_databases')['reuters']

    for f in files:
        file = (folder_path + '/' + f)
        if f[-7:] == '.df.pkl': # NORMAL
            print(file)
            with open((file[:-24] + '.json')) as f:
                json_data = json.load(f)
            df = pd.read_pickle(file, compression='zip')
            df = df_rename(df=df, json_data=json_data)

            if df is not None:
                df['request_id'] = df['request_id'] + offset_request_id
                df['TR_REVENUE_DATE'] = pd.to_datetime(df['TR_REVENUE_DATE'], format='%Y-%m-%d')
                df['TR_BSPERIODENDDATE'] = pd.to_datetime(df['TR_BSPERIODENDDATE'], format='%Y-%m-%d')

                with my.postgresql_connect(param_dic) as conn:
                    df_insert_sql(conn=conn, df=df, table='data')
                    update_request_status(request_id=df['request_id'].unique().tolist(), conn=conn, new_status=1)
            else:
                with my.postgresql_connect(param_dic) as conn:
                    request_id = int(str(file).split('_')[-3][-6:]) + offset_request_id
                    print('No data for request', request_id)
                    update_request_status(request_id=request_id, conn=conn, new_status=1)
                    error_dict = {'task_id': request_id, 'timestamp_iso': datetime.datetime.now().isoformat(), 'error_type': 'LOGICAL', 'error_code': 'Reuters output is all None'}
                    submit_error(error_dict=error_dict, conn=conn)
        elif file[-8:] == '.err.pkl': # ERROR
            print(file)
            object_file = pickle.load(open(file, 'rb'))
            error_dict = dict(object_file)
            error_dict['task_id'] = error_dict['task_id'] + offset_request_id
            with my.postgresql_connect(param_dic) as conn:
                update_request_status(request_id=error_dict['task_id'], conn=conn, new_status=-1)
                submit_error(error_dict=error_dict, conn=conn)





if __name__ ==  '__main__':
    qrt_folder_path = '/Users/vanalmsick/Workspace/MasterThesis/app/get_data/reuters_eikon/portable_app/2021-04-17_09-15-30 data'
    ann_folder_path = '/Users/vanalmsick/Workspace/MasterThesis/app/get_data/reuters_eikon/portable_app/2021-04-17_00-18-22 data'

    data_to_sql(qrt_folder_path, offset_request_id=-7988)
    data_to_sql(ann_folder_path, offset_request_id=-7988+217)