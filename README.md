# MasterThesis
Predicting company earnings (Net income / ROE / dividends per share) with AI from company fundamentals (B/S, I/S, CF/S, ShrE/S).

## Data-Flow/Process-Graphic
The approach is to apply best practices with the most widely used (and best considered) software and tools:
![data flow chart](resources/data_flow_chart.png "data flow chart")

## 1. Getting the data *([folder: a](app/a_get_data))*
**Two methods:**
- **Big-Data-Method:** Downloading all data fields filled >= 66% form Reuters Eikon & WRDS Compustat
- **Handpicked-Method:** Download all key metrics/data fields specified in csv-file and intelligently fill NaNs with formulas and methods.  
  
Please see the [README_get_data.md](app/a_get_data/README_get_data.md) for detailed instructions/overview.  

## 2. Data cleaning and missing data handling *([folder: b](app/b_data_cleaning))*
Intelligent NaN filling *([see code here](app/b_data_cleaning/data_cleaning.py))* with formulas and methods specified in last columns beginning with "fillnan_" in [this csv-file](app/a_get_data/reuters_eikon/key_reuters_fields.csv).  

**Example for intelligent NaN-filling formulas and methods: *([see csv-file](app/a_get_data/reuters_eikon/key_reuters_fields.csv))***  

| Category | Name | Reuters Code | Clear Name | fillnan_1 | fillnan_2 |
| ------------- |:-------------:| -----:| -----:| -----:| -----:|
| General | Data date | TR.TotalReturn.date | data_date | | |
| Income Statement | Depreciation and amortization | TR.DepreciationDepletion+TR.Amortization | DeprArmo | formula:EBITDA-EBIT | formula:Depreciation+Amortization |
| Income Statement | EBIT | TR.EBIT | EBIT | formula:EBITDA-DeprArmo |  |
| Profitability | Tax Rate | TR.IncomeTaxRatePct | TaxRate | formula:Tax/PreTaxIncome*100 | method:linear(or=mean) |
| Ratios | Asset Turnover | TR.F.AssetTurnover | AssetTurnover | formula:Sales/TotalAssets |  |
  
**Available fillnan-methods are:**
- value (e.g. value:0 -> fill NaNs with 0)
- formula (e.g. formula:Sales/TotalAssets with the column names from the "Clear Name"-column)
- method
   - linear(or=mean) with or can be any number or "mean"
   - approx(other=COLUMN_NAME) with other being one or more other columns used to linearly approx that column e.g. approx(other=Sales) or approx(other=\[Sales, EBIT])
   
**To run this code: *([see code here](app/b_data_cleaning/data_cleaning.py))***
```python
## Get cleaned and filled data
# General params
recache_raw_data = False
redo_data_cleaning = False

# Dataset params: Option 1 manual props/overwriting props
data_version = 'handpicked_dataset'  # Get name from app/b_data_cleaning/_dataset_registry.py file
comp_col = 'ric'
time_cols = ['data_year', 'data_qrt']
industry_col = 'industry'

# Dataset params: Option 2 get props from registry
data_version = 'handpicked_dataset'  # Get name from app/b_data_cleaning/_dataset_registry.py file
from app.b_data_cleaning import get_dataset_registry
dataset_props = get_dataset_registry()[data_version]
comp_col = dataset_props['company_col']
time_cols = dataset_props['iter_cols']
industry_col = dataset_props['industry_col']

# Data dropping params
required_filled_cols_before_filling = ['sales']  # drop rows before NaN filling if columns NaN/not filled
required_filled_cols_after_filling = ['ebit', 'roe']  # drop rows after NaN filling if columns NaN/not filled
drop_threshold_row_pct = 0.25  # drop rows before NaN filling if columns of row are filled less than percent
drop_threshold_row_quantile = 0.2  # drop rows before NaN filling if columns of row are filled less than quantile percentage
drop_threshold_col_pct = 0  # drop columns before NaN filling if column is less percent filled
append_data_quality_col = False  # append 'data_quality' column with percentage of columns filled per row before NaN filling

# Run code
from app.b_data_cleaning.data_cleaning import get_clean_data
df_cleaned = get_clean_data(data_version, recache_raw_data=recache_raw_data, redo_data_cleaning=redo_data_cleaning, comp_col=comp_col, time_cols=time_cols, industry_col=industry_col, required_filled_cols_before_filling=required_filled_cols_before_filling, required_filled_cols_after_filling=required_filled_cols_after_filling, drop_threshold_row_pct=drop_threshold_row_pct, drop_threshold_row_quantile=drop_threshold_row_quantile, drop_threshold_col_pct=drop_threshold_col_pct, append_data_quality_col=append_data_quality_col)
print(df_cleaned)
```  

## 3. Data prep / feature engerneering *([folder: c](app/c_data_prep))*
- Feature engerneeing (ratios / pct-change / feature selection / etc. - [see code here](app/c_data_prep/i_feature_engineering.py))
```python
from app.c_data_prep.i_feature_engineering import feature_engerneeing
# Give cleaned data from (1.)
comp_col = 'ric'
time_cols = ['data_year', 'data_qrt']
industry_col = 'industry'
df_feature_engineered = feature_engerneeing(dataset=df_cleaned, comp_col=comp_col, time_cols=time_cols, industry_col=industry_col, recache=False)
```
- Data normalization (mean-std dev normalization on block/time-step/compayn-time-step-level - [see code here](app/c_data_prep/ii_data_prep.py))
- Data sperating into training sets (caching final datasets for training to increase speed when training and feature turning - [see code here](app/c_data_prep/ii_data_prep.py))
```python
from app.c_data_prep.ii_data_prep import data_prep
# Give cleaned and feature engineered data from step before (2.1.)
y_cols = ['roe']
time_cols = ['data_year', 'data_qrt']
comp_col = 'ric'
keep_raw_cols = []  # columns to not normalize
drop_cols = []  # columns to drop 

# Initialize data-class
data = data_prep(dataset=df_feature_engineered, y_cols=y_cols, iter_cols=time_cols, comp_col=comp_col, keep_raw_cols=keep_raw_cols, drop_cols=drop_cols)

# Define rolling window size
data.window(input_width=5*4, pred_width=4, shift=1)

# Select one from three split methods
#data.single_time_rolling(val_time_steps=1, test_time_steps=1, shuffle=True)
data.block_rolling_split(val_comp_size=0, test_comp_size=0, train_time_steps=5*4*2, val_time_steps=1, test_time_steps=1, shuffle=True)
#data.block_static_split(val_comp_size=0, test_comp_size=0, val_time_size=0.2, test_time_size=0.1, shuffle=True)

# Select one from three normalization levels
#data.normalize(method='block')
data.normalize(method='time')
#data.normalize(method='set')

# Compute/cache the datsets
data.compute()

# Get one time-step dataset
out = data['200201_201903']
ds_train, ds_val, ds_test = data.tsds_dataset(out='all', out_dict=None)

# Export last time-step dataset to excel to inspect
data.export_to_excel()
```
  
## 4. Prediction *([folder: d](app/d_prediction))*
**prediction.py** ([in app/d_prediction](app/d_prediction/prediction.py)) executes all models:
1. baseline models *([see code here](app/d_prediction/baseline_models.py))*
    - Baseline_last_value: prediction = last quarter value
    - Baseline_4last_value: prediction = value 4 quarters ago
2. statistical models *([see code here](app/d_prediction/statistical_models.py))*
    - linear
    - multi-dense: multiple linear layers
    - ToDo: logistic model
3. Advanced Statisctical / ML models *([see code here](app/d_prediction/ML_xxx_models.py))*
    - ToDo: ARIMA / SARIMA
    - ToDo: Holt-Winters
    - ToDo: Ranodm Forest/decision Tree
    - ToDo: XGBoost/LightGBM
4. Deep Leraning / Neural Networks *([see code here](app/d_prediction/NN_tensorflow_models.py))*
    - LSTM
    - ??? MLP / CNN / RNN / GRU
```python
# ToDo: Add code snip hwo to execute when all models coded and optimal parameters choosen
```