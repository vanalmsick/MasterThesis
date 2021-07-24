# MasterThesis
Predicting company earnings (Net income / ROE / dividends per share) with AI from company fundamentals (B/S, I/S, CF/S, ShrE/S).

## 1. Getting the data *(folder: a)*
Please see the [README_get_data.md](app/a_get_data/README_get_data.md)

## 2. Data cleaning and missing data handling *(folder: b)*
Intelligent NaN filling.
  
## 3. Data prep / feature engerneering *(folder: c)*
- Feature engerneeing
- Data normalization
- Data sperating into training sets
  
## 4. Prediction *(folder: d)*
prediction.py (in app/d_prediction) executes all models:
1. baseline models
    - Baseline_last_value: prediction = last quarter value
    - Baseline_4last_value: prediction = value 4 quarters ago
2. statistical models
    - linear
    - multi-dense: multiple linear layers
    - ToDo: logistic model
3. Advanced Statisctical / ML models
    - ToDo: ARIMA / SARIMA
    - ToDo: Holt-Winters
    - ToDo: Ranodm Forest/decision Tree
    - ToDo: XGBoost/LightGBM
4. Deep Leraning / Neural Networks
    - LSTM
    - ??? MLP / CNN / RNN / GRU