# MasterThesis
Predicting company earnings (Net income / ROE / dividends per share) with AI from company fundamentals (B/S, I/S, CF/S, ShrE/S).

## 1. Getting the data *(folder: a)*
Please see the [README_get_data.md](app/a_get_data/README_get_data.md)  
  
**Two methods:**
- **Big-Data-Method:** Downloading all data fields filled >= 66% form Reuters Eikon & WRDS Compustat
- **Handpicked-Method:** Download all key metrics/data fields specified in csv-file and intelligently fill NaNs with formulas and methods.

## 2. Data cleaning and missing data handling *(folder: b)*
Intelligent NaN filling *([see code here](app/b_data_cleaning/data_cleaning.py))* with formulas and methods specified in last columns beginning with "fillnan_" in [this csv-file](app/a_get_data/reuters_eikon/key_reuters_fields.csv).
  
## 3. Data prep / feature engerneering *(folder: c)*
- Feature engerneeing (ratios / pct-change / feature selection / etc. - [see code here](app/c_data_prep/i_feature_engineering.py))
- Data normalization (mean-std dev normalization on block/time-step/compayn-time-step-level - [see code here](app/c_data_prep/ii_data_prep.py))
- Data sperating into training sets (caching final datasets for training to increase speed when training and feature turning - [see code here](app/c_data_prep/ii_data_prep.py))
  
## 4. Prediction *(folder: d)*
prediction.py ([in app/d_prediction](app/d_prediction/prediction.py)) executes all models:
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