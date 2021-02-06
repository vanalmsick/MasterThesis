import psycopg2, psycopg2.extras, warnings, sys, ast, datetime
import pandas as pd



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


def sql_query(sql_query, conn):
    table = pd.read_sql_query(sql_query, conn)
    return table


def df_insert_sql(conn, df, table):
    """
    Using psycopg2.extras.execute_values() to insert the dataframe
    """
    # Create a list of tupples from the dataframe values
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
    print("execute_values() done")
    cursor.close()


def reuters_eikon_data_scraper(instruments: list, fields: list, properties:dict, api_key: str, test=False):
    if test:
        str_date = datetime.datetime.strptime('2020-12-31','%Y-%m-%d')
        df = pd.DataFrame({'Date':[str_date, str_date, str_date, str_date, str_date],'Symbol': ['APPL','MSFT','MMM','COKE','VW'], 'Revenue':[1.5, 3.2, 8.4, 3.9, 5.1], 'GrossProfit':[10.1, 20.5, 7.3, 50.4, 9.2]}, index=[0,1,2,3,4])
    else:
        import eikon as ek
        ek.set_app_key(api_key)

        # Examples:
        # instruments = ['GOOG.O', 'MSFT.O', 'FB.O', 'AMZN.O', 'TWTR.K'],
        # fields = ['TR.Revenue.date','TR.Revenue','TR.GrossProfit'],
        # properties = {'Scale': 6, 'SDate': 0, 'EDate': -2, 'FRQ': 'FY', 'Curn': 'EUR'}

        df, err = ek.get_data(instruments, fields, properties)

        if err is not None:
            warnings.warn(err)

    return df



def df_to_key_value_pairs(df, idx_cols, col_dict):
    clean_col_dict = col_dict.copy()
    for k in idx_cols:
        clean_col_dict.pop(k, None)

    df.rename(columns=idx_cols, inplace=True)
    df.rename(columns=col_dict, inplace=True)

    loop_cols = df.columns.to_list()
    data_labels = [x for x in loop_cols if x not in list(idx_cols.values())]

    new_df = pd.DataFrame()
    col_list = list(idx_cols.values())
    col_list.extend(['data_label', 'data_value'])

    for _, row in df.iterrows():
        tmp = row.copy()
        for data_label in data_labels:
            tmp['data_label'] = data_label
            tmp['data_value'] = tmp[data_label]
            new_df = new_df.append(tmp[col_list])
    return new_df


def main():
    false_data = True  # FOR DEVELOPMENT
    sql_password = str(input('Please enter the password for the SQL Database:'))
    reuters_api = str(input('Please enter the Reuters Eikon API Key:'))


    param_dic = {
        "host": "master-thesis.cx5hfb0s7vvw.eu-central-1.rds.amazonaws.com",
        "port": "5432",
        "database": "postgres",
        "user": "vanalmsick",
        "password": sql_password
    }

    with postgresql_connect(param_dic) as conn:
        SQL = "SELECT * FROM getdata_todo"
        task_list = sql_query(SQL, conn)
        task_list.set_index('id', inplace=True)

        SQL = "SELECT reuters_name, clear_name FROM data_fields"
        data_fields = sql_query(SQL, conn)
        data_fields.set_index('clear_name', inplace=True)
        data_fields = data_fields.to_dict()['reuters_name']

    idx_cols = {'Date': 'ref_date', 'Symbol': 'stock_symbol', 'report_type':'report_type', 'request_id':'request_id'}

    try:
        conn = postgresql_connect(param_dic)

        for id, row in task_list.iterrows():
            instruments = ast.literal_eval(row['instruments'])
            fields = ast.literal_eval(row['fields'])
            properties = ast.literal_eval(row['properties'])

            data = reuters_eikon_data_scraper(instruments=instruments, fields=fields, properties=properties, api_key=reuters_api, test=false_data)

            if properties['FRQ'] == 'FY':
                data['report_type'] = 1
            elif properties['FRQ'] == 'QRT':
                data['report_type'] = 4
            else:
                raise Exception('UNKNOWN report_type', properties['FRQ'])
            data['request_id'] = id

            key_value = df_to_key_value_pairs(df=data, idx_cols=idx_cols, col_dict=data_fields)
            now = datetime.datetime.now()
            now = now.isoformat()
            key_value['last_updated'] = now
            answ = df_insert_sql(conn, key_value, 'data')

            if answ[0] == 1:
                # Error
                status = -1
                sql = """INSERT INTO getdata_errors (timestamp, todo_id, error_message) VALUES ('{}', {}, '{}')""".format(now, id, str(answ[1]))
                with conn.cursor() as cur:
                    cur.execute(sql)
            else:
                # Success
                status = 1
            sql = """UPDATE getdata_todo SET status = {}, timestamp = '{}' WHERE id = {}""".format(status, now, id)
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            #print(key_value)
    finally:
        conn.close()





if __name__=='__main__':
    main()