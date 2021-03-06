import pandas as pd
import numpy as np
import wrds
import eikon as ek
import sys, os, ast, progressbar, time, warnings


### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import z_helpers as my
##################################






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

    len_todos = len(reuters_todos) + len(wrds_todos)
    j = 0

    if (len(reuters_todos) + len(wrds_todos)) != 0:

        # Progressbar to see hwo long it will still take
        print('\nGetting Data from Reuters and WRDS and save it:')
        time.sleep(0.5)
        widgets = ['[',
                   progressbar.Timer(format='elapsed: %(elapsed)s'),
                   '] ',
                   progressbar.Bar('█'), ' (',
                   progressbar.ETA(), ') ',
                   ]
        progress_bar = progressbar.ProgressBar(max_value=len_todos, widgets=widgets).start()

        with my.postgresql_connect(aws_param) as aws_conn:

            if len(reuters_todos) > 0:
                # Do Reuters To Do s
                first_item = True
                for request_id, row in reuters_todos.iterrows():
                    instruments = ast.literal_eval(row['req_instruments'])
                    fields = ast.literal_eval(row['req_fields'])
                    properties = ast.literal_eval(row['req_parameters'])
                    df, err = my.reuters_eikon_data_scraper(instruments=instruments, fields=fields, properties=properties, api_key=reuters_api_key)

                    if err is None:
                        # push df
                        df['request_id'] = request_id
                        df['task_id'] = row['task_id']
                        df['report_type'] = properties['FRQ']
                        if first_item:
                            my.create_sql_tbl(df, conn=aws_conn, tbl_name='data_statements', schema='reuters')
                            first_item = False
                        my.df_insert_sql(conn=aws_conn, df=df, table='reuters.data_statements', schema='reuters')
                        my.update_request_status(request_id=request_id, conn=aws_conn, schema='reuters', new_status=1)
                    else:
                        # repot error
                        my.update_request_status(request_id=request_id, conn=aws_conn, schema='reuters', new_status=-1)
                        err['request_id'] = request_id
                        my.submit_error(error_dict=err, conn=aws_conn, schema='reuters')
                        max_len = min(len(err['error_message']), 40)
                        warnings.warn('Reuters request {} returned an {}-Error with the message: {}'.format(request_id, err['error_type'], err['error_message'][:max_len]))

                    j += 1
                    progress_bar.update(j)


            if len(wrds_todos) > 0:
                # Do WRDS to Do s
                first_item = True
                with wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password) as wrds_conn:
                    for request_id, row in wrds_todos.iterrows():

                        df, err = my.wrds_compustat_data_scraper(conn=wrds_conn, sql=row['req_sql'])

                        if err is None:
                            # push df
                            df['request_id'] = request_id
                            df['task_id'] = row['task_id']
                            if row['target_table'] == 'data_statements':
                                df['report_type'] = row['req_table']
                            if first_item:
                                my.create_sql_tbl(df, conn=aws_conn, tbl_name=row['target_table'], schema='wrds')
                                first_item = False
                            if len(df) > 0:
                                my.df_insert_sql(conn=aws_conn, df=df, table=row['target_table'], schema='wrds')
                            my.update_request_status(request_id=request_id, conn=aws_conn, schema='wrds', new_status=1)
                        else:
                            # repot error
                            my.update_request_status(request_id=request_id, conn=aws_conn, schema='reuters', new_status=-1)
                            err['request_id'] = request_id
                            my.submit_error(error_dict=err, conn=aws_conn, schema='reuters')
                            max_len = min(len(err['error_message']), 40)
                            warnings.warn('Reuters request {} returned an {}-Error with the message: {}'.format(request_id, err['error_type'], err['error_message'][:max_len]))

                        j += 1
                        progress_bar.update(j)






if __name__ == '__main__':
    my.convenience_settings()

    get_data()  # The big all in dataset

    print('Everything done! No Open ToDos left.')