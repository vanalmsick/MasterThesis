import pandas as pd
import os, warnings, datetime, progressbar, time, ast, pickle, json, sys, traceback


def reuters_eikon_data_scraper(instruments: list, fields: list, properties:dict, api_key: str, id, test=False):
    if test:
        str_date = datetime.datetime.strptime('2020-12-31','%Y-%m-%d')
        data_df = pd.DataFrame({'Date':[str_date, str_date, str_date, str_date, str_date],'Symbol': ['APPL','MSFT','MMM','COKE','VW'], 'Revenue':[1.5, 3.2, 8.4, 3.9, 5.1], 'GrossProfit':[10.1, 20.5, 7.3, 50.4, 9.2]}, index=[0,1,2,3,4])
        data_json = {}
        data_json['people'] = []
        data_json['people'].append({'name': 'Scott', 'website': 'stackabuse.com', 'from': 'Nebraska'})
        data_json['people'].append({'name': 'Larry', 'website': 'google.com', 'from': 'Michigan'})
        data_json['people'].append({'name': 'Tim', 'website': 'apple.com', 'from': 'Alabama'})
        error = None
    else:
        import eikon as ek
        ek.set_app_key(api_key)

        # Examples:
        # instruments = ['GOOG.O', 'MSFT.O', 'FB.O', 'AMZN.O', 'TWTR.K'],
        # fields = ['TR.Revenue.date','TR.Revenue','TR.GrossProfit'],
        # properties = {'Scale': 6, 'SDate': 0, 'EDate': -2, 'FRQ': 'FY', 'Curn': 'EUR'}
        #print(instruments, fields, properties)
        #print(type(instruments), type(fields), type(properties))
        data_df, data_json, error = None, None, None
        now = datetime.datetime.now()
        now_iso_str = now.isoformat()
        try:
            data_json = ek.get_data(instruments, fields, properties, raw_output=True)
            data_df, err = ek.get_data(instruments, fields, properties, raw_output=False)

            if err is not None:
                error = {'timestamp': now, 'timestamp_iso': now_iso_str, 'task_id': id,
                         'reuters_request': {'instruments': instruments, 'fields': fields, 'properties': properties},
                         'error_type': 'REUTERS', 'error_code': err}
                warnings.warn('REUTERS ERROR with task {} getting data for {}.'.format(id, instruments))
            else:
                data_df['report_type'] = properties['FRQ']
                data_df['request_id'] = id
                now = datetime.datetime.now()
                now_iso = now.isoformat()
                data_df['last_updated'] = now_iso
        except Exception as err:
            exc_info = sys.exc_info()
            error = {'timestamp': now, 'timestamp_iso': now_iso_str, 'task_id': id,
                     'reuters_request': {'instruments': instruments, 'fields': fields, 'properties': properties},
                     'error_type': 'PYTHON', 'error_code': str(err).replace('\n',';').encode()}
            warnings.warn('PYTHON ERROR with taks {} getting message: {}'.format(id, str(err)))
            pass

    return data_df, data_json, error





def main_data_scraper(reuters_api, todo_list_path, output_folder, false_dev_data=False):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    task_list = pd.read_csv(todo_list_path, sep=';', header=0, index_col=0, parse_dates=[-1])
    task_list.sort_index(inplace=True)

    # Progressbar to see hwo long it will still take
    print('\nGetting Data from Eikon and saving it:')
    time.sleep(0.5)
    widgets = ['[',
               progressbar.Timer(format='elapsed: %(elapsed)s'),
               '] ',
               progressbar.Bar('█'), ' (',
               progressbar.ETA(), ') ',
               ]
    progress_bar = progressbar.ProgressBar(max_value=len(task_list), widgets=widgets).start()


    j = 0
    for id, row in task_list.iterrows():
        instruments = ast.literal_eval(row['instruments'])
        fields = ast.literal_eval(row['fields'])
        fields.append('TR.Revenue.date')
        properties = ast.literal_eval(row['properties'])

        data_df, data_json, err = reuters_eikon_data_scraper(instruments=instruments, fields=fields, properties=properties, api_key=reuters_api, test=false_dev_data, id=id)

        id_prefix = f"{j:04d}" + f"_id-{int(id):06d}"
        if err is not None:
            pickle.dump(err, open((output_folder + id_prefix + ".err.pkl"), "wb"))
        else:
            data_df.to_pickle((output_folder + id_prefix + '_reuters_data-zip.df.pkl'), compression='zip')
            with open((output_folder + id_prefix + ".json"), 'w') as outfile:
                json.dump(data_json, outfile)

        j += 1

        progress_bar.update(j)


if __name__ == '__main__':
    reuters_api = "f48ad3b9259a47a9b9359198fc164cb9ce3e2085"

    #os.chdir("J:\WinPy\portable_app")
    print('Wokring dir:', os.getcwd())

    todo_list_path = 'portable_app/data_todo_list.csv'
    print('ToDo-List file:', todo_list_path)
    output_folder = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S") + ' data/'
    print('Data Output dir:', output_folder)

    false_dev_data = False
    main_data_scraper(reuters_api=reuters_api, todo_list_path=todo_list_path, output_folder=output_folder, false_dev_data=false_dev_data)
