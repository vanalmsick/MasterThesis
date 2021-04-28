import pandas as pd
import numpy as np
import wrds
import eikon as ek
import sys, os, ast, progressbar, time, warnings


### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import helpers as my
##################################


def reuters_eikon_data_scraper(instruments: list, fields: list, properties:dict, api_key:str):
    df, err = None, None
    ek.set_app_key(api_key)
    try:
        df, err = ek.get_data(instruments, fields, properties, field_name=True, raw_output=False)
        if err is not None:
            err = {'error_type': 'REUTERS', 'error_message': str(err), 'req_instruments': instruments, 'req_fields': fields, 'req_properties': properties}
        else:
            df.columns = [i.replace('.', '_') for i in df.columns.tolist()]
    except Exception as error:
        err = {'error_type': 'PyREUTERS', 'error_message': str(error).replace('\n', ';').encode(), 'req_instruments': instruments, 'req_fields': fields, 'req_properties':properties}
    finally:
        return df, err


def wrds_compustat_data_scraper(conn, sql):
    df, err = None, None
    try:
        df = conn.raw_sql(sql)
    except Exception as error:
        err = {'error_type': 'PyWRDS', 'error_message': str(error).replace('\n', ';').encode(), 'req_sql': sql}
    finally:
        return df, err



def get_data():
    aws_param = my.get_credentials(credential='aws')
    reuters_api_key = my.get_credentials(credential='reuters_eikon_api')
    wrds_username = my.get_credentials(credential='wrds_credentials')['username']
    wrds_password = my.get_credentials(credential='wrds_credentials')['password']

    # Get To Do s
    with my.postgresql_connect(aws_param) as conn:
        reuters_todos = my.sql_query(sql="SELECT * FROM reuters.request_todos", conn=conn)
        wrds_todos = my.sql_query(sql="SELECT * FROM wrds.request_todos", conn=conn)
    reuters_todos.set_index('request_id', inplace=True)
    wrds_todos.set_index('request_id', inplace=True)

    max_reuters_id = 0 if np.isnan(reuters_todos.index.max()) else reuters_todos.index.max()
    max_wrds_id = 0 if np.isnan(wrds_todos.index.max()) else wrds_todos.index.max()

    if (max_wrds_id + max_reuters_id) != 0:

        # Progressbar to see hwo long it will still take
        print('\nGetting Data from Reuters and WRDS and save it:')
        time.sleep(0.5)
        widgets = ['[',
                   progressbar.Timer(format='elapsed: %(elapsed)s'),
                   '] ',
                   progressbar.Bar('█'), ' (',
                   progressbar.ETA(), ') ',
                   ]
        progress_bar = progressbar.ProgressBar(max_value=(max_reuters_id + max_wrds_id), widgets=widgets).start()

        with my.postgresql_connect(aws_param) as aws_conn:

            if len(reuters_todos) > 0:
                # Do Reuters To Do s
                first_item = True
                for request_id, row in reuters_todos.iterrows():
                    instruments = ast.literal_eval(row['req_instruments'])
                    fields = ast.literal_eval(row['req_fields'])
                    properties = ast.literal_eval(row['req_parameters'])
                    df, err = reuters_eikon_data_scraper(instruments=instruments, fields=fields, properties=properties, api_key=reuters_api_key)

                    if err is None:
                        # push df
                        df['request_id'] = request_id
                        df['task_id'] = row['task_id']
                        df['report_type'] = properties['FRQ']
                        if first_item:
                            my.create_sql_tbl(df, conn=aws_conn, tbl_name='data_statements', schema='reuters')
                            first_item = False
                        my.df_insert_sql(conn=aws_conn, df=df, table='reuters.data_statements')
                        my.update_request_status(request_id=request_id, conn=aws_conn, schema='reuters', new_status=1)
                    else:
                        # repot error
                        my.update_request_status(request_id=request_id, conn=aws_conn, schema='reuters', new_status=-1)
                        err['request_id'] = request_id
                        my.submit_error(error_dict=err, conn=aws_conn, schema='reuters')
                        max_len = min(len(err['error_message']), 40)
                        warnings.warn('Reuters request {} returned an {}-Error with the message: {}'.format(request_id, err['error_type'], err['error_message'][:max_len]))

                    progress_bar.update(request_id)


            if len(wrds_todos) > 0:
                # Do WRDS to Do s
                first_item = True
                with wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password) as wrds_conn:
                    for request_id, row in wrds_todos.iterrows():

                        df, err = wrds_compustat_data_scraper(conn=wrds_conn, sql=row['req_sql'])

                        if err is None:
                            # push df
                            df['request_id'] = request_id
                            df['task_id'] = row['task_id']
                            df['report_type'] = row['req_table']
                            if first_item:
                                my.create_sql_tbl(df, conn=aws_conn, tbl_name='data_statements', schema='wrds')
                                first_item = False
                            my.df_insert_sql(conn=aws_conn, df=df, table='wrds.data_statements')
                            my.update_request_status(request_id=request_id, conn=aws_conn, schema='wrds', new_status=1)
                        else:
                            # repot error
                            my.update_request_status(request_id=request_id, conn=aws_conn, schema='reuters', new_status=-1)
                            err['request_id'] = request_id
                            my.submit_error(error_dict=err, conn=aws_conn, schema='reuters')
                            max_len = min(len(err['error_message']), 40)
                            warnings.warn('Reuters request {} returned an {}-Error with the message: {}'.format(request_id, err['error_type'], err['error_message'][:max_len]))

                        progress_bar.update((max_reuters_id + request_id))




if __name__ == '__main__':
    my.convenience_settings()
    get_data()

    print('Everything done! No Open ToDos left.')