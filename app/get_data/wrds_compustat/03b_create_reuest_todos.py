import os, sys, datetime
import pandas as pd


### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import helpers as my
##################################


def create_todo_list():
    aws_param = my.get_credentials(credential='aws')

    with my.postgresql_connect(aws_param) as conn:
        df_rics = my.sql_query(sql="SELECT * FROM wrds.gvkey_list", conn=conn)
    company_list = df_rics['gvkey'].tolist()

    n = 2
    task_id = 1
    date_iso = datetime.datetime.now().isoformat()

    company_list_of_list = [company_list[i:i + n] for i in range(0, len(company_list), n)]
    columns = '*'
    req_table = 'fundq'

    new_todos = []
    for companies in company_list_of_list:
        conditions = {'gvkey': companies}
        custom_condition = "datadate > '1995-01-01'"
        out_condition = {'custom_condition': custom_condition, 'conditions': conditions}
        sql_query = my.create_wrds_sql_query(table=req_table, columns=columns, distinct=False, conditions=conditions, custom_condition=custom_condition, no_observations=-1)
        new_todos.append([task_id, str(columns), str(req_table), str(out_condition), str(sql_query), 0, date_iso])
    new_todos = pd.DataFrame(new_todos, columns=['task_id', 'req_columns', 'req_table', 'req_conditions', 'req_sql', 'status', 'last_updated'])


    with my.postgresql_connect(aws_param) as conn:
        my.df_insert_sql(conn, df=new_todos, table='wrds.data_request_list')


if __name__ == '__main__':
    create_todo_list()