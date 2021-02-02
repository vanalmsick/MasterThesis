#######################################################################################################################
# Gets data from Reuter's Eikon via Python package 'eikon'.
#######################################################################################################################

# Import helpers
import sys, os, warnings
current_dir = os.path.dirname(os.path.abspath(os.getcwd()))
helpers_dir = os.path.join(current_dir, 'helpers')
sys.path.append(helpers_dir)
import helpers

import urllib, requests
import lxml.html as lh
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


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


def get_reuters_ric(search_list):
    url_left_part = 'https://www.reuters.com/finance/stocks/lookup?searchType=any&comSortBy=marketcap&sortBy=&dateRange=&search='

    dict = {}

    for search_term in search_list:
        try:
            print('Getting RIC of "{}"...'.format(search_term))
            url_right_part = urllib.parse.quote(search_term)
            url = url_left_part + url_right_part
            print(url)

            page = requests.get(url)
            soup = BeautifulSoup(page.text,'lxml')
            serach_table = soup.find('table', {'class': 'search-table-data'})
            df = pd.read_html(str(serach_table))[0]

            dict[search_term] = df['Symbol'].loc[0]
            print('RIC for "{}" is "{}".'.format(search_term, dict[search_term]))
        except:
            print('ERROR: Was not able to get RIC for "{}".'.format(search_term))
            dict[search_term] = ''

    return dict


def remove_empty_lines(filename):
    if not os.path.isfile(filename):
        print("{} does not exist ".format(filename))
        return
    with open(filename) as filehandle:
        lines = filehandle.readlines()

    with open(filename, 'w') as filehandle:
        lines = filter(lambda x: x.strip(), lines)
        filehandle.writelines(lines)



def main(**kwargs):

    if not os.path.isfile('reuters_ric_list.txt'):
        # Get list of S&P 500
        sp500_list = pd.read_csv('S&P 500 list.csv', sep=',', header=0, index_col=False)

        # Get Reuters RIC ID from Name
        reuters_ric_list = get_reuters_ric(sp500_list['Name'].to_list())

        # Save Mapping
        pd.DataFrame.from_dict(reuters_ric_list, orient='index').to_csv('ric_mapping.csv', header=False)

        # Save Reuters RIC list
        with open('reuters_ric_list.txt', 'w') as f:
            f.write('\n'.join(list(reuters_ric_list.values())))
        remove_empty_lines('reuters_ric_list.txt')



    reuters_api_key = helpers.get_credentials(cred='reuters_eikon_api')
    database = helpers.database('income_statement')

    with open('datapoint_list.txt', 'r') as f:
        data_fields = [i.rstrip('\n') for i in f.readlines()]
    with open('reuters_ric_list.txt', 'r') as f:
        ric_list = [i.rstrip('\n') for i in f.readlines()]
    ric_chunks = [ric_list[x:x + 5] for x in range(0, len(ric_list), 100)]

    properties = {'Scale': 6, 'SDate': 0, 'EDate': -10, 'FRQ': 'FY', 'Curn': 'USD'}

    for instruments in ric_chunks:
        df = reuters_eikon_data_scraper(instruments=instruments, fields=data_fields, properties=properties, api_key=reuters_api_key)
        database.append(df)
        print(database.get())




if __name__ == '__main__':
    main()


