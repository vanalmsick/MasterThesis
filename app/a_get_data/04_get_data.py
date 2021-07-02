import pandas as pd
import numpy as np
import wrds
import eikon as ek
import sys, os, ast, progressbar, time, warnings


### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import z_helpers as my
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
                    df, err = reuters_eikon_data_scraper(instruments=instruments, fields=fields, properties=properties, api_key=reuters_api_key)

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

                        df, err = wrds_compustat_data_scraper(conn=wrds_conn, sql=row['req_sql'])

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



def get_handpicked_dataset():
    years_back = 27
    reuters_max_data_limit = 50000

    aws_param = my.get_credentials(credential='aws')

    data_dir = my.get_project_directories(key='data_dir')
    company_list_file = os.path.join(data_dir, 'index_constituents.csv')
    company_df = pd.read_csv(company_list_file)
    company_list = company_df['constituent_ric'].to_list()

    col_df = pd.read_csv('a_get_data/reuters_eikon/key_reuters_fields.csv')
    col_df = col_df[col_df['Data Type'] == 'time series']
    col_list = col_df['Reuters Code'].to_list()

    companies_per_request = int((reuters_max_data_limit / len(col_list)) / (years_back * 4)) - 1
    company_list_of_lists = instrument_list_of_list = [company_list[i:i + companies_per_request] for i in range(0, len(company_list), companies_per_request)]

    col_rename_dict = col_df[['Reuters Code', 'Clear Name']].set_index('Reuters Code')['Clear Name'].to_dict()
    col_rename_dict = dict(zip([i.upper() for i in list(col_rename_dict.keys())], list(col_rename_dict.values())))
    col_rename_dict['INSTRUMENT'] = 'ric'
    app_dict = {}
    for key, value in col_rename_dict.items():
        app_dict[key.replace('.','_')] = value
    col_rename_dict.update(app_dict)

    id = 900000

    # Progressbar to see hwo long it will still take
    print('\nGetting Small dataset from Reuters and save it:')
    time.sleep(0.5)
    widgets = ['[',
               progressbar.Timer(format='elapsed: %(elapsed)s'),
               '] ',
               progressbar.Bar('█'), ' (',
               progressbar.ETA(), ') ',
               ]
    progress_bar = progressbar.ProgressBar(max_value=len(company_list_of_lists) + 1, widgets=widgets).start()


    first_item = True
    with my.postgresql_connect(aws_param) as aws_conn:
        for comps in company_list_of_lists:
            id += 1
            df, err = reuters_eikon_data_scraper(instruments=comps,
                                                 fields=col_list,
                                                 properties={"SDate": 0, "EDate": -4 * years_back, "Frq": "CQ", "Curn": "USD"},
                                                 api_key=my.get_credentials(credential='reuters_eikon_api'))
            #print('Error:', err)

            df.columns = df.columns.str.upper()
            df = df.rename(columns=col_rename_dict)
            df = df[(df['Date'].notna() & ~df['Date'].isnull() & (df['Date'] != ''))]
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df = df.replace('', np.nan, regex=True)
            df = df.replace('NaN', np.nan, regex=True)
            df = df.replace('Infinity', np.nan, regex=True)
            df = df.replace('-Infinity', np.nan, regex=True)
            numeric_cols = col_df[~col_df['Variable Type'].isin(['property', 'category'])]['Clear Name'].tolist()
            df[numeric_cols] = df[numeric_cols].astype(float)


            if df is not None and len(df) > 0:
                if first_item:
                    my.create_sql_tbl(df, conn=aws_conn, tbl_name='data_small', schema='reuters')
                    first_item = False
                my.df_insert_sql(conn=aws_conn, df=df, table='data_small', schema='reuters')
            if err is not None:
                err['request_id'] = id
                my.submit_error(error_dict=err, conn=aws_conn, schema='reuters')

            progress_bar.update(id - 900000)








if __name__ == '__main__':
    my.convenience_settings()

    #get_data()  # The big all in dataset
    get_handpicked_dataset()  # The small reuters dataset

    print('Everything done! No Open ToDos left.')