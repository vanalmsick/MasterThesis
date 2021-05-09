import os, sys


### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import helpers as my
##################################


def constitutents_get(indices):
    import eikon as ek
    api_key = my.get_credentials(credential='reuters_eikon_api')
    ek.set_app_key(api_key)

    instruments = list(indices.keys())
    fields = ['TR.IndexName',
            'TR.IndexJLConstituentChangeDate',
            'TR.IndexJLConstituentRIC',
            'TR.IndexJLConstituentName',
            'TR.IndexJLConstituentComName'
            ]

    data_df, err = ek.get_data(instruments, fields=fields, parameters={'SDate': '1995-01-01', 'EDate': '2021-04-15'}, raw_output=False)
    data_df.columns = ['Index_RIC'] + [i.replace('.', '_') for i in fields]
    data_df['TR_IndexName'] = data_df['Index_RIC'].map(indices)
    data_df['Constituent_RIC'] = [i[0] for i in data_df['TR_IndexJLConstituentRIC'].str.split('^').tolist()]

    data_dir = my.get_project_directories(key='data_dir')
    output_file_path = os.path.join(data_dir, 'index_constituents.csv')
    data_df.columns = [x.lower() for x in data_df.columns]
    data_df.to_csv(output_file_path, index=False)



if __name__ == '__main__':
    index_dict = {'.SPX': 'S&P 500', '.DJI': 'DOW JONES INDUSTRIAL AVERAGE', '.NDX':'NASDAQ 100', '.IXIC': 'NASDAQ COMPOSITE'}

    constitutents_get(indices=index_dict)