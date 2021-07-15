# Get Data
*Back to [Main README.md](../../README.md)*  
  
There are two methods implemented to get data:
1. Big-Data-Method
2. Csv-Handpicked-Method

## 1. Big-Data-Method
First **all index constitutents** are fetched from reuters by exectuting:
- reuters_eikon/01_get_index_constituents.py -> csv-file  

constitutents_get(indices=my_indices)  

my_indices = {'.SPX': 'S&P 500', '.DJI': 'DOW JONES INDUSTRIAL AVERAGE', '.NDX':'NASDAQ 100', '.IXIC': 'NASDAQ COMPOSITE'}

The index constitutent list is then imported to a SQL-database.

Then the **reuters-instruemnt-identifiers ("ric"s) are matched with the wrds/compustat "gvkey"s** by exetuting:
- wrds_compustat/02_get_gvkey_mapping.py

get_gvkey(input from SQL-table) -> csv gvkey-ric mapping file

This mapping is also imported to a SQL-database.

Then **data-request ToDos are created** using splitting the data in smaller chunks:
- reuters_eikon/03a_create_request_todos.py (reuters) with data-fields requested hard-coded in varaible "fields" (every Balance-Sheet/Income-Statement/CF-Statement reuters fields that was filled >= 66% for the above index constitutents)
- wrds_compustat/03b_create_request_todos.py (wrds/compustat) with getting every possible data-field
=> ToDo-lists in SQL-table

Then the data **request ToDos will be executed** sequentially by executing:
- 04a_get_big_dataset.py

## 2. Handpicked-Dataset
Just a small handpicked list of data-fields is downloaded fromreuters eikon.  
  
First, the **list of data-fields** to get is defined:
- reuters_eikon/key_reuters_fields.csv

Second, getting a list of **index-constitutuents** by exectuting:
- reuters_eikon/01_get_index_constituents.py -> csv-file  

constitutents_get(indices=my_indices)  

my_indices = {'.SPX': 'S&P 500', '.DJI': 'DOW JONES INDUSTRIAL AVERAGE', '.NDX':'NASDAQ 100', '.IXIC': 'NASDAQ COMPOSITE'}


Then, these for the above companies the above specified data-fields are **downloaded from reuters and imported to an SQL-table** by exectuting:
- 04b_get_handpicked_dataset.py