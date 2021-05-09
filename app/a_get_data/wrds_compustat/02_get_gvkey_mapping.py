import os, sys
import wrds
from fuzzy_match import algorithims
import pandas as pd
import numpy as np


### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import helpers as my
##################################





def get_gvkey():
    aws_param = my.get_credentials(credential='aws')

    with my.postgresql_connect(aws_param) as conn:
        df_companies = my.sql_query(sql="SELECT * FROM reuters.company_list", conn=conn)
    company_list = [i.split('.')[0] for i in df_companies['constituent_ric'].tolist()]


    wrds_username = my.get_credentials(credential='wrds_credentials')['username']
    wrds_password = my.get_credentials(credential='wrds_credentials')['password']
    wrds_conn = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)

    gvkey_mapping = wrds_conn.raw_sql(my.create_wrds_sql_query(table='security', columns=['tic','gvkey'], distinct=True, conditions={'tic': company_list}, no_observations=-1))
    gvkey_list = gvkey_mapping['gvkey'].tolist()


    wrds_companies = wrds_conn.raw_sql(my.create_wrds_sql_query(table='company', columns=['gvkey', 'conm', 'conml', 'fic', 'loc', 'weburl'], distinct=True, conditions={'gvkey': gvkey_list}, no_observations=-1))


    df_companies['tic'] = company_list
    df_combined = pd.merge(df_companies, gvkey_mapping, on='tic', how='left')
    df_combined = pd.merge(df_combined, wrds_companies, on='gvkey', how='left')
    df_combined['match'] = df_combined.apply(lambda x: algorithims.levenshtein(x['tr_indexjlconstituentcomname'], x['conml']), axis=1)
    df_combined['match'].fillna(0, inplace=True)
    df_combined['row_add'] = [i / 1000000 for i in list(range(len(df_combined), 0, -1))]
    df_combined['score'] = df_combined['match'] + df_combined['row_add']
    grouped_df = df_combined.groupby('score')
    maxs = grouped_df.max()
    df_combined = maxs.reset_index()
    df_combined = df_combined[['constituent_ric', 'tic', 'gvkey']]
    df_combined['include_company'] = np.where(df_combined['gvkey'].isna(), False, True)

    data_dir = my.get_project_directories(key='data_dir')
    output_file_path = os.path.join(data_dir, 'reuters_wrds_mapping.csv')
    df_combined.to_csv(output_file_path, index=False)






if __name__ == '__main__':
    get_gvkey()