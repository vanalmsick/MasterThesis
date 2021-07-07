import pandas as pd
import numpy as np
import wrds
import eikon as ek
import sys, os, ast, progressbar, time, warnings


### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import z_helpers as my
##################################




def get_handpicked_dataset():
    years_back = 27
    reuters_max_data_limit = 50000

    aws_param = my.get_credentials(credential='aws')

    data_dir = my.get_project_directories(key='data_dir')
    company_list_file = os.path.join(data_dir, 'index_constituents.csv')
    company_df = pd.read_csv(company_list_file)
    company_list = company_df['constituent_ric'].unique().tolist()

    col_raw_df = pd.read_csv('a_get_data/reuters_eikon/key_reuters_fields.csv')
    props_df = col_raw_df[col_raw_df['Data Type'] == 'property']
    col_df = col_raw_df[col_raw_df['Data Type'] == 'time series']
    props_list = props_df['Reuters Code'].to_list()
    col_list = col_df['Reuters Code'].to_list()

    companies_per_request = max(int((reuters_max_data_limit / len(col_list)) / (years_back * 4)) - 1, 4)
    company_list_of_lists = instrument_list_of_list = [company_list[i:i + companies_per_request] for i in range(0, len(company_list), companies_per_request)]

    col_rename_dict = col_raw_df[['Reuters Code', 'Clear Name']].set_index('Reuters Code')['Clear Name'].to_dict()
    col_rename_dict = dict(zip([i.upper() for i in list(col_rename_dict.keys())], list(col_rename_dict.values())))
    col_rename_dict['INSTRUMENT'] = 'ric'
    app_dict = {}
    for key, value in col_rename_dict.items():
        app_dict[key.replace('.','_')] = value
    col_rename_dict.update(app_dict)

    id = 900000


    first_item = True
    with my.postgresql_connect(aws_param) as aws_conn:

        # General company props (like name, indust
        print('Getting company properties...')
        df, err = my.reuters_eikon_data_scraper(instruments=company_list, fields=props_list, properties={"Curn": "USD"},
                                             api_key=my.get_credentials(credential='reuters_eikon_api'))
        df.columns = df.columns.str.upper()
        df = df.rename(columns=col_rename_dict)
        df['filled_sorter'] = df.count(axis=1)
        df = df.sort_values('filled_sorter', ascending=False)
        df = df.drop_duplicates(subset=['ric'], keep='first')
        df.drop(columns=['filled_sorter'], inplace=True)
        df = df.sort_values('ric', ascending=True)
        my.create_sql_tbl(df, conn=aws_conn, tbl_name='data_small_props', schema='reuters')
        my.df_insert_sql(conn=aws_conn, df=df, table='data_small_props', schema='reuters')
        print('Got company properties.')


        # Progressbar to see hwo long it will still take
        print('\nGetting Small quaterly dataset from Reuters and save it:')
        time.sleep(0.5)
        widgets = ['[',
                   progressbar.Timer(format='elapsed: %(elapsed)s'),
                   '] ',
                   progressbar.Bar('█'), ' (',
                   progressbar.ETA(), ') ',
                   ]
        progress_bar = progressbar.ProgressBar(max_value=len(company_list_of_lists) + 1, widgets=widgets).start()


        # Get yearly data
        for comps in company_list_of_lists:
            id += 1
            df, err = my.reuters_eikon_data_scraper(instruments=comps,
                                                 fields=col_list,
                                                 properties={"SDate": 0, "EDate": -4 * years_back, "Frq": "CQ", "Curn": "USD"},
                                                 api_key=my.get_credentials(credential='reuters_eikon_api'))
            #print('Error:', err)



            if df is not None:
                df = df.dropna(how='all', subset=df.columns.to_list()[1:])  # drop empty rows
                if len(df) > 0:
                    df.columns = df.columns.str.upper()
                    df = df.rename(columns=col_rename_dict)
                    df = df[(df['data_date'].notna() & ~df['data_date'].isnull() & (df['data_date'] != ''))]

                    df = df.replace(r'^\s*$', np.nan, regex=True)
                    df = df.replace('', np.nan, regex=True)
                    df = df.replace('NaN', np.nan, regex=True)
                    df = df.replace('Infinity', np.nan, regex=True)
                    df = df.replace('-Infinity', np.nan, regex=True)
                    numeric_cols = col_df[~col_df['Variable Type'].isin(['property', 'category'])]['Clear Name'].tolist()
                    df[numeric_cols] = df[numeric_cols].astype(float)

                    df['request_id'] = id
                    df['task_id'] = 900000
                    df['report_type'] = "CQ"

                    if len(df) > 0:
                        # Remove duplicates
                        df['filled_sorter'] = df.count(axis=1)
                        df = df.sort_values('filled_sorter', ascending=False)

                        df['data_year'] = df['data_date'].str[:4].astype(int)
                        df['data_month'] = df['data_date'].str[5:7].astype(int)
                        df['data_day'] = df['data_date'].str[8:10].astype(int)
                        df['data_qrt'] = df['data_month'] // 3
                        df['year_qrt_id'] = (df['data_year'] - 1950) * 4 + df['data_qrt']

                        df = df.drop_duplicates(subset=['ric', 'data_year', 'data_qrt'], keep='first')
                        df.drop(columns=['filled_sorter'], inplace=True)
                        df = df.sort_values(['ric', 'year_qrt_id'], ascending=True)
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

    get_handpicked_dataset()  # The small reuters dataset

    print('Got all data.')