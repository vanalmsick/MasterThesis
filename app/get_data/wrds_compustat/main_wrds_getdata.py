import pandas as pd
import numpy as np
import wrds
import psycopg2, psycopg2.extras, sys


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


def dtype_mapping():
    return {'object' : 'TEXT',
        'int64' : 'INT',
        'float64' : 'FLOAT',
        'datetime64' : 'DATETIME',
        'bool' : 'TINYINT',
        'category' : 'TEXT',
        'timedelta[ns]' : 'TEXT'}


def gen_tbl_cols_sql(df):
    dmap = dtype_mapping()
    sql = "pi_db_uid SERIAL"
    df1 = df.rename(columns = {"" : "nocolname"})
    hdrs = df1.dtypes.index
    hdrs_list = [(hdr, str(df1[hdr].dtype)) for hdr in hdrs]
    for i, hl in enumerate(hdrs_list):
        sql += " ,{0} {1}".format(hl[0], dmap[hl[1]])
    return sql


def create_sql_tbl(df, conn, tbl_name):
    tbl_cols_sql = gen_tbl_cols_sql(df)
    sql = "CREATE TABLE {0} ({1})".format(tbl_name, tbl_cols_sql)
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()
    conn.commit()


def sql_query(sql_query, conn):
    table = pd.read_sql_query(sql_query, conn)
    return table

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

def sep_list_chunks(l, n):
    final = [l[i:min(i + n, len(l))] for i in range(0, len(l), n)]
    return final

def dev_test():
    wrds_username = "USERNAME"
    wrds_password = "PASSWORD"

    param_dic = {
        "host": "localhost",
        "port": "5432",
        "database": "wrds_compustat",
        "user": "postgres",
        "password": "PASSWORD"
    }

    columns = '*'
    library = 'comp'
    table = 'company' # fundq # funda  # company
    obs = -1
    offset = 0

    if obs < 0:
        obsstmt = ''
    else:
        obsstmt = ' LIMIT {}'.format(obs)
    if columns is None:
        cols = '*'
    else:
        cols = ','.join(columns)

    sql = """SELECT DISTINCT gvkey FROM companies"""
    conn = postgresql_connect(param_dic)
    gvkey_master_list = sql_query(sql, conn)['gvkey'].to_list()

    gvkey_master_list = sep_list_chunks(l=gvkey_master_list, n=4)

    wrds_conn = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)

    for gvkey_list in gvkey_master_list:
        print(gvkey_list)
        orig_list = gvkey_list
        gvkey_list = [str(i).zfill(6) for i in gvkey_list]
        gvkey_list = "', '".join(gvkey_list)

        #  AND datadate > '1998-01-01'
        sql = ("SELECT {cols} FROM {schema}.{table} WHERE gvkey in ('{gvkey_list}') {obsstmt} OFFSET {offset};".format(
            cols=cols,
            schema=library,
            table=table,
            obsstmt=obsstmt,
            gvkey_list=gvkey_list,
            offset=offset))
        print(sql)

        #print(conn.list_tables(library='comp'))
        company = wrds_conn.raw_sql(sql)
        company.rename(columns={'do':'do_'}, inplace=True)

        if orig_list == gvkey_master_list[0]:
            create_sql_tbl(company, conn, 'comp_props')

        df_insert_sql(conn, company, 'comp_props')

        print(company)

    wrds_conn.close()
    conn.close()




if __name__ == '__main__':
    dev_test()