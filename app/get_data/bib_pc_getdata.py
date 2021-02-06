import pyodbc, warnings
import pandas as pd



def sql_connect(server_address, server_port, database_name, user_name, user_password):
    conn = pyodbc.connect('DRIVER={};Server={};Port={};Database={};User ID={};Password={};String Types=Unicode'.format('Devart ODBC Driver for PostgreSQL', server_address, server_port, database_name, user_name, user_password))
    return conn


def reuters_eikon_data_scraper(instruments: list, fields: list, properties:dict, api_key: str):
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



def main():
    connection = sql_connect("master-thesis.cx5hfb0s7vvw.eu-central-1.rds.amazonaws.com", "5432", "postgres", "vanalmsick", "zuVxyg-porxax-0zawzu")
    SQL = "SELECT * FROM getdata_todo"
    result = pd.read_sql(SQL, connection)
    print(result)



if __name__=='__main__':
    main()