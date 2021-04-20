import datetime

import pandas as pd
import numpy as np
import pickle
import os, json, psycopg2, psycopg2.extras, sys, math


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
    np_array = df.replace({pd.np.nan: None}).to_numpy()
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


def df_rename(df, head):
    col_list = df.columns.tolist()
    col_should_be = [i['displayName'] for i in head]
    added_cols = ['report_type', 'request_id', 'last_updated']
    col_should_be = col_should_be + added_cols
    new_col_list = ['stock'] + [i['field'].replace('.','_') for i in head[1:]] + added_cols
    if (col_list == col_should_be) == False:
        print(len(new_col_list), len(col_list))
        print([i for i in new_col_list if i not in col_list])
        print([i for i in col_list if i not in new_col_list])
        raise Exception('df does not have columns as expected')
    df.columns = new_col_list
    return df


def data_to_sql(folder_path):
    files = os.listdir(folder_path)

    param_dic = my.get_credentials(credential='local_databases')['reuters']

    for f in files:
        file = (folder_path + '/' + f)
        print(file)
        if f[-7:] == '.df.pkl': # NORMAL
            with open((file[:-24] + '.json')) as f:
                json_data = json.load(f)
            head_col = json_data['headers'][0]
            df = pd.read_pickle(file, compression='zip')
            df = df_rename(df=df, head=head_col)

            df['request_id'] = df['request_id'] - 7988 # + 217
            df['TR_REVENUE_DATE'] = pd.to_datetime(df['TR_REVENUE_DATE'], format='%Y-%m-%d')
            df['TR_BSPERIODENDDATE'] = pd.to_datetime(df['TR_BSPERIODENDDATE'], format='%Y-%m-%d')

            with my.postgresql_connect(param_dic) as conn:
                df_insert_sql(conn=conn, df=df, table='data')
        elif os.path.isfile((file[:-24] + '.err.pkl')): # ERROR
            pkl_file = (file[:-24] + '.err.pkl')
            #print(pkl_file)
            object_file = pickle.load(open(pkl_file, 'rb'))
            #print(object_file)



def error_to_sql(folder_path):
    pass


if __name__ ==  '__main__':
    qrt_folder_path = '/Users/vanalmsick/Workspace/MasterThesis/app/get_data/reuters_eikon/portable_app/2021-04-17_09-15-30 data'
    #ann_folder_path = '/Users/vanalmsick/Workspace/MasterThesis/app/get_data/reuters_eikon/portable_app/2021-04-17_00-18-22 data'

    data_to_sql(qrt_folder_path)