import psycopg2, psycopg2.extras, warnings, sys, ast, datetime, time, os, zipfile, shutil
import pandas as pd
import numpy as np


### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import helpers as my
##################################




def df_insert_sql(conn, df, table):
    """
    Using psycopg2.extras.execute_values() to insert the dataframe
    """
    # Create a list of tupples from the dataframe values
    df = df.where(pd.notnull(df), None)
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
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

def sql_query(sql_query, conn):
    table = pd.read_sql_query(sql_query, conn)
    return table

class data_label():
    def __init__(self):
        self.big_task_id = None

    def translate(self, df, param_dic, big_task_id):
        if big_task_id == self.big_task_id:
            translate_dict = self.translate_dict
        else:
            translate_dict = {}

        self.big_task_id = big_task_id
        clear_name_list = list(df['data_label'].replace("'","").unique())


        j = 0
        for clear_name in clear_name_list:
            if clear_name not in translate_dict:
                sql = """SELECT DISTINCT reuters_name FROM data_fields WHERE clear_name = '{}'""".format(str(clear_name).replace("'",""))
                with my.postgresql_connect(param_dic) as conn:
                    reuters_name = sql_query(sql, conn)
                if len(reuters_name) == 1:
                    translate_dict[clear_name] = reuters_name['reuters_name'].iloc[0]
                elif len(reuters_name) == 2:
                    sql = """SELECT reuters_name, count(id) as counter FROM data_fields WHERE clear_name = '{}' GROUP BY reuters_name""".format(str(clear_name).replace("'",""))
                    with my.postgresql_connect(param_dic) as conn:
                        reuters_name = sql_query(sql, conn)
                    if reuters_name['counter'].iloc[0] > reuters_name['counter'].iloc[1] + 2:
                        translate_dict[clear_name] = str(reuters_name['reuters_name'].iloc[0])
                    elif reuters_name['counter'].iloc[1] > reuters_name['counter'].iloc[0] + 2:
                        translate_dict[clear_name] = str(reuters_name['reuters_name'].iloc[1])
                    else:
                        translate_dict[clear_name] = 'UNKNOWN(2):' + clear_name
                        j += 1
                else:
                    translate_dict[clear_name] = 'UNKNOWN(?):' + clear_name
                    j += 1

        df['data_label'].replace(translate_dict, inplace=True)
        print('{} UNKONW'.format(str(j)))
        self.translate_dict = translate_dict
        return df





def main_import_tosql():
    import_path = '/data/import_getdata'
    file_list = [f for f in os.listdir(import_path) if os.path.isfile(os.path.join(import_path, f))]
    param_dic = my.get_credentials(credential='local_databases')['postgres']

    file_list.remove('.DS_Store')
    tranlate = data_label()

    for file in file_list:
        print('File:', file)
        with zipfile.ZipFile(os.path.join(import_path, file), 'r') as zip_ref:
            zip_ref.extractall(import_path)
        time.sleep(1)
        shutil.move(os.path.join(import_path, file), os.path.join(os.path.join(import_path, 'archive'), file))

    file_list = [f for f in os.listdir(import_path) if os.path.isfile(os.path.join(import_path, f))]
    file_list.remove('.DS_Store')

    for file in file_list:
        print('######################### {} #########################'.format(file))
        csv_name = file.split('.')[0] + '.csv'
        df = pd.read_csv(os.path.join(import_path, csv_name), index_col=False, header=0, sep=';')
        df = tranlate.translate(df, param_dic, 1)
        print(df)


        with my.postgresql_connect(param_dic) as conn:
            df_insert_sql(conn=conn, df=df, table='data')

        now = datetime.datetime.now()
        now = now.isoformat()
        id_list = list(df['request_id'].astype(int).unique())
        for id in id_list:
            print(id)
            sql = """UPDATE getdata_todo SET status = {}, timestamp = '{}' WHERE id = {}""".format(str(1), now, id)
            with my.postgresql_connect(param_dic) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                conn.commit()
        os.remove(os.path.join(import_path, csv_name))









if __name__ == '__main__':
    main_import_tosql()