import os, sys, zlib, hashlib, warnings, math, datetime, random
import psycopg2, psycopg2.extras, pickle, progressbar, time, hashlib
import pandas as pd
import numpy as np
import tensorflow as tf

# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')
from app.z_helpers import helpers as my
from app.b_data_cleaning.data_cleaning import get_clean_data


def get_pct_change(df, time_cols=['period_year', 'period_qrt'], company_cols=['gvkey'], reset_index=True):
    df_cp = df.copy()
    df_cp = df_cp.set_index(company_cols + time_cols)
    df_cp = df_cp.sort_index()
    df_cp = df_cp.select_dtypes(np.number).groupby(level=company_cols).pct_change()
    if reset_index:
        df_cp = df_cp.reset_index()
    return df_cp.fillna(0)

def get_shifted_df(df, shift=1, time_cols=['period_year', 'period_qrt'], company_cols=['gvkey'], reset_index=True):
    df_cp = df.copy()
    df_cp = df_cp.set_index(company_cols + time_cols)
    df_cp = df_cp.sort_index()
    df_cp = df_cp.select_dtypes(np.number).groupby(level=company_cols).shift(shift)
    if reset_index:
        df_cp = df_cp.reset_index()
    return df_cp



def industry_average(df, time_cols=['period_year', 'period_qrt'], industry_cols=['iid'], reset_index=True):
    df_cp = df.copy()
    df_cp = df_cp.groupby(industry_cols + time_cols)
    df_cp = df_cp.mean()
    if reset_index:
        df_cp = df_cp.reset_index()
    return df_cp


def compare_against_industry(df_comps, df_industry, time_cols, company_col, industry_col, as_int=False):
    comb = df_comps.merge(df_industry, on=time_cols + [industry_col], how="left", suffixes=("_comp", "_ind"))
    cols = df_comps.columns.tolist()
    cols = [i for i in cols if i in df_industry.columns.tolist() and i not in time_cols and i != industry_col]
    new_df = comb[time_cols + [company_col]].copy()
    for col in cols:
        if pd.api.types.is_numeric_dtype(comb[col + '_comp']):
            if as_int:
                new_df[col] = (comb[col + '_comp'] >= comb[col + '_ind']).astype(int)
            else:
                new_df[col] = (comb[col + '_comp'] >= comb[col + '_ind'])
        else:
            new_df[col] = np.nan
    return new_df




def lev_thiagaranjan_signs(df_abs, df_pct, iter_col, company_col):
    new_df = df_pct[[company_col] + iter_col].copy()
    new_df['1_inventory'] = df_pct['inventory'].fillna(0) - df_pct['sales']
    #new_df['2_accounts receivals'] = df_pct['receivablesandloanstotal'] - df_pct['sales']
    new_df['3_cap exp'] = df_pct['capitalexpenditures_avg'] - df_pct['capitalexpenditures']
    new_df['4_RnD'] = df_pct['randd_avg'] - df_pct['randd']
    new_df['5_gross margin'] = df_pct['sales'] - df_pct['grossmargin']
    new_df['6_sales admin exp'] = df_pct['sga'] - df_pct['sales']
    # ToDo: Prov doubt accounts no/little data
    #new_df['7_prov doubt rec'] = df_pct['receivables'] - df_pct['provdoubtacct']
    # ToDo: 8_eff tax: df['tr_f_ebit'] tr_f_inctaxratepct
    new_df['9_order backlog'] = df_pct['sales'] - df_pct['orderbacklog'].fillna(0)
    new_df['10_labor force'] = ((df_abs['sales_sft_1'] / df_abs['employeenum_sft_1']) - (df_abs['sales'] / df_abs['employeenum'])) / df_abs['sales_sft_1'] / df_abs['employeenum_sft_1']
    new_df.loc[df_abs['fifovslifo'] < 0, '11_FIFO dummy'] = 1
    new_df['11_FIFO dummy'].fillna(0, inplace=True)
    new_df.loc[df_abs['fifovslifo'] > 0, '11_LIFO dummy'] = 1
    new_df['11_LIFO dummy'].fillna(0, inplace=True)
    # ToDo: 12_Audit dummy

    return new_df



def ou_pennmann_signs(df_abs, df_pct, iter_col, company_col):
    new_df = df_pct[[company_col] + iter_col].copy()
    new_df['2_current ratio'] = df_pct['currentratio']  # .fillna(0)
    new_df['4_quick ratio'] = df_pct['quickratio']
    new_df['8_inventory turnover'] = df_pct['inventoryturnover']
    new_df['9_inventory to assets'] = df_abs['inventorytoassets']
    new_df['10_inventory to assets pct'] = df_pct['inventorytoassets']
    new_df['11_inventory'] = df_pct['inventory']
    new_df['12_sales'] = df_pct['sales']
    new_df['13_depreciation'] = df_pct['depreciation']
    new_df['14_div per share'] = df_abs['dividendperstock'] - df_abs['dividendperstock_sft_1']
    # ToDo:DeprToPPEsomehwo missing in data
    #new_df['16_depr to ppe'] = df_pct['deprttoppe']
    new_df['17_ROE'] = df_abs['roe']
    new_df['18_ROE pct chg'] = df_pct['roe']
    new_df['19_CAPEX To Assets'] = df_pct['capextoassets']
    new_df['20_CAPEX To Assets last year'] = df_pct['capextoassets_sft_4']
    new_df['21_debt to equity'] = df_abs['debttoequity']
    new_df['22_debt to equity pct chg'] = df_pct['debttoequity']
    new_df['30_sales to assets'] = df_abs['salestoassets']
    new_df['31_ROA'] = df_abs['roa']
    new_df['33_gross margin'] = df_abs['grossmargin']
    new_df['38_pretax income to sales'] = df_pct['pretaxincometosales']
    new_df['41_sales to total cash'] = df_abs['salestocash']
    new_df['53_total assets'] = df_pct['totalassets']
    new_df['54_CF to debt'] = df_abs['fcftodebt']
    new_df['55_WC to assets'] = df_abs['wctoassets']
    new_df['57_OpIncome to assets'] = df_abs['opinctoassets']
    new_df['61_Repayment of LT debt '] = df_abs['ltdebtrepaypct']
    new_df['66_Cash div to cash flows'] = df_abs['divtofcf']

    return new_df


def xue_zhang_signs(df_abs, df_pct, iter_col, company_col, industry_col):
    new_df = df_abs[[company_col] + [industry_col] + iter_col].copy()
    # ToDo: Binarize by  comparing against industry avg
    new_df['1_profit margin'] = df_pct['netprofitmargin']
    new_df['2_ROA'] = df_abs['roa']
    new_df['3_ROA chg'] = df_pct['roa']
    new_df['4_CF to Assets'] = df_abs['leveredfocf'] / df_abs['totalassets']
    new_df['5_Accruals'] = df_abs['assetaccruals']
    new_df['6_Accounts Receivable Turnover'] = df_pct['acctrcvblturnover']
    new_df['7_Inventory Turnover'] = df_pct['inventoryturnover']
    new_df['8_Asset Turnover'] = df_pct['assetturnover']
    new_df['9_Current Ratio'] = df_pct['currentratio']
    new_df['10_Quick Ratio'] = df_pct['quickratio']
    new_df['11_Working Capital'] = df_pct['wc']

    print('xue_zhang', new_df.notna().mean() * 100)

    df_ind_avg = industry_average(new_df, time_cols=iter_col, industry_cols=[industry_col], reset_index=True)

    dummy_comp_vs_ind = compare_against_industry(df_comps=new_df, df_industry=df_ind_avg, time_cols=iter_col, company_col=company_col, industry_col=industry_col, as_int=True)

    return dummy_comp_vs_ind


def dummy_signs(df, dummy_cols, iter_col, company_col):
    new_df = pd.get_dummies(df, prefix=dummy_cols, columns=dummy_cols, dummy_na=True, drop_first=True)
    drop_cols = [i for i in df.columns.tolist() if i not in [company_col] + iter_col]
    new_df.drop(columns=drop_cols, inplace=True, errors='ignore')

    return new_df



def y_columns(df_abs, df_pct, iter_col, company_col, industry_col, y_cols='all', drop_y_chg_col=True):
    new_df = df_abs[[company_col] + [industry_col] + iter_col].copy()
    #new_df['t_year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    #new_df['t_year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    new_df['y_eps'] = df_abs['eps']
    new_df['y_eps pct'] = df_pct['eps']
    new_df['y_dividendyield'] = df_abs['dividendyield']
    new_df['y_dividendyield pct'] = df_pct['dividendyield']
    new_df['y_dividendperstock'] = df_abs['dividendperstock']
    new_df['y_dividendperstock pct'] = df_pct['dividendperstock']
    new_df['y_roe'] = df_abs['roe']
    new_df['y_roe pct'] = df_pct['roe']
    new_df['y_roa'] = df_abs['roa']
    new_df['y_roa pct'] = df_pct['roa']
    new_df['y_EBIT'] = df_abs['ebit']
    new_df['y_EBIT pct'] = df_pct['ebit']
    new_df['y_Net income'] = df_abs['netincome']
    new_df['y_Net income pct'] = df_pct['netincome']


    new_df.drop(columns=[industry_col], inplace=True)

    if y_cols == 'all':
        return new_df
    else:
        return new_df[[company_col] + iter_col + y_cols]



def merge_df(index_cols, merge_dfs):
    new_df = merge_dfs[0].copy()
    for i in range(1, len(merge_dfs)):
        new_df = pd.merge(new_df, merge_dfs[i], how='inner', on=index_cols)
    return new_df


def _reduce_qrt_to_yrl(df, col='eps'):

    df.set_index(['ric'] + ['data_year', 'data_qrt'], inplace=True)
    df.sort_index(inplace=True)
    df['y_pct_chg_ident'] = df[col].groupby(level=['ric']).shift(1)
    df.reset_index(inplace=True)
    df[['t_chg', 'y_std']] = np.nan

    for name, row in df.groupby('ric'):
        tmp_1 = row.loc[row['data_qrt'] == 1, 'y_pct_chg_ident'].std()
        tmp_2 = row.loc[row['data_qrt'] == 2, 'y_pct_chg_ident'].std()
        tmp_3 = row.loc[row['data_qrt'] == 3, 'y_pct_chg_ident'].std()
        tmp_4 = row.loc[row['data_qrt'] == 4, 'y_pct_chg_ident'].std()
        tmp_max = max(tmp_1, tmp_2, tmp_3, tmp_4)
        tmp_dict = {tmp_1:1, tmp_2:2, tmp_3:3, tmp_4:4}
        df.loc[df['ric']==name, 't_chg'] = tmp_dict[tmp_max]
        df.loc[df['ric']==name, 'y_std'] = tmp_max

    df_all = df[df['data_qrt'] == df['t_chg']]

    #df_all['t_i_qrt_sin'] = np.sin((df_all['t_chg']) * np.pi / 2) #.astype(int)
    #df_all['t_i_qrt_cos'] = np.cos((df_all['t_chg']) * np.pi / 2) #.astype(int)


    df_all.drop(columns=['y_std', 'y_pct_chg_ident'], inplace=True)
    df_all['data_qrt'] = 0

    return df_all


def _lagged_variables(df, lagged_dict={'__all__': [1, 2, 3, 4]}, comp_col=['ric'], time_cols=['data_year', 'data_qrt'], exclude_cols=[]):
    if '__all__' in lagged_dict:
        all_cols = df.columns.tolist()
        all_cols = [col for col in all_cols if col not in lagged_dict and col not in comp_col and col not in time_cols and col not in exclude_cols]
        for col in all_cols:
            lagged_dict[col] = lagged_dict['__all__']
        lagged_dict.pop('__all__')

    for col in exclude_cols:
        lagged_dict.pop(col, None)

    df.set_index(comp_col + time_cols, inplace=True)
    df.sort_index(inplace=True)

    for col, shifts in lagged_dict.items():
        for shift in shifts:
            df[col + f'_sft_{shift}'] = df[col].groupby(level=['ric']).shift(shift)

    df.reset_index(inplace=True)

    return df




def feature_engerneeing(dataset, comp_col, time_cols, industry_col, all_features='all', yearly_data=False):

    df = dataset

    df['t_qrt_sin'] = np.sin((df['data_qrt'] - 1) * np.pi / 2).astype(int)
    df['t_qrt_cos'] = np.cos((df['data_qrt'] - 1) * np.pi / 2).astype(int)

    if yearly_data:
        df = _reduce_qrt_to_yrl(df, col='eps')

    df_ind_avg = industry_average(df, time_cols=time_cols, industry_cols=[industry_col], reset_index=True)
    df = df.merge(df_ind_avg[time_cols + [industry_col] + ['capitalexpenditures', 'randd']], how='left', on=time_cols + [industry_col], validate='many_to_one', suffixes=['', '_avg'])
    df_shift = get_shifted_df(df, shift=1, time_cols=time_cols, company_cols=[comp_col], reset_index=True).fillna( method='bfill')
    df = df.merge(df_shift[time_cols + [comp_col] + ['sales', 'employeenum', 'dividendperstock']], how='left', on=time_cols + [comp_col], validate='many_to_one', suffixes=['', '_sft_1'])

    df_pct = get_pct_change(df, time_cols=time_cols, company_cols=[comp_col], reset_index=True)
    df_pct_shift4 = get_shifted_df(df_pct, shift=4, time_cols=time_cols, company_cols=[comp_col], reset_index=True).fillna(method='bfill')
    df_pct = df_pct.merge(df_pct_shift4[time_cols + [comp_col] + ['capextoassets']], how='left', on=time_cols + [comp_col], validate='many_to_one', suffixes=['', '_sft_4'])

    df_lev_thi = lev_thiagaranjan_signs(df_abs=df, df_pct=df_pct, iter_col=time_cols, company_col=comp_col)
    df_ou_penn = ou_pennmann_signs(df_abs=df, df_pct=df_pct, iter_col=time_cols, company_col=comp_col)
    df_xue_zha = xue_zhang_signs(df_abs=df, df_pct=df_pct, iter_col=time_cols, company_col=comp_col, industry_col=industry_col)

    df_dummy = dummy_signs(df=df, dummy_cols=[industry_col, 'sector', 'exchangename', 'headquarterscountry', 'analystrecom'], iter_col=time_cols, company_col=comp_col)

    df_y = y_columns(df_abs=df, df_pct=df_pct, iter_col=time_cols, company_col=comp_col, industry_col=industry_col, drop_y_chg_col=yearly_data == False)

    dfs = []
    if all_features == 'all' or 'lev_thi' in all_features:
        dfs.append(df_lev_thi)
    if all_features == 'all' or 'ou_penn' in all_features:
        dfs.append(df_ou_penn)
    if all_features == 'all' or 'xue_zha' in all_features:
        dfs.append(df_xue_zha)
    dfs.append(df_y)


    df_all = merge_df(index_cols=time_cols + [comp_col], merge_dfs=dfs)

    df_all = _lagged_variables(df_all, lagged_dict={'__all__': [1, 2, 3, 4]}, comp_col=[comp_col],
                           time_cols=time_cols, exclude_cols=[industry_col, 'sector', 'exchangename', 'headquarterscountry', 'analystrecom'])

    df_all = df_all.replace(['inf', 'nan', '-inf', np.inf, -np.inf, np.nan], np.nan)

    # ToDo: What do in the end if NaN after feature engerneeing? drop or fill?
    df_all = df_all.dropna()

    # Drop outlioers
    from scipy import stats
    cols = df_all.columns.to_list()
    for col in df_all.columns.to_list():
        if df_all[col].dtype == object:
            cols.remove(col)
            print('not use col', col)
        else:
            max = df_all[col].max()
            min = df_all[col].min()
            if max == 1 and min == -1 or max == 1 and min == 0 or max == 0 and min == 0 or max == 1 and min == 1:
                cols.remove(col)
                print('not use col', col)
    for col in time_cols:
        if col in cols:
            cols.remove(col)
    df_z_check = df_all[cols]
    keep_rows = (np.abs(stats.zscore(df_z_check.replace(['inf', 'nan', np.inf, -np.inf, np.nan], 0))) < 3).all(axis=1)

    df_all = df_all[keep_rows]

    return df_all



if __name__ == '__main__':
    # Working directory must be the higher .../app folder
    from app.z_helpers import helpers as my
    my.convenience_settings()

    dataset_name = 'handpicked_dataset'

    from app.b_data_cleaning import get_dataset_registry
    dataset_props = get_dataset_registry()[dataset_name]

    recache_raw_data = False
    redo_data_cleaning = False

    comp_col = dataset_props['company_col']
    time_cols = dataset_props['iter_cols']
    industry_col = dataset_props['industry_col']

    required_filled_cols_before_filling = ['sales', 'roe', 'ebit']
    required_filled_cols_after_filling = []
    drop_threshold_row_pct = 0.4
    drop_threshold_row_quantile = 0.15
    drop_threshold_col_pct = 0
    append_data_quality_col = False

    df_cleaned = get_clean_data(dataset_name, recache_raw_data=recache_raw_data, redo_data_cleaning=redo_data_cleaning,
                                comp_col=comp_col, time_cols=time_cols, industry_col=industry_col,
                                required_filled_cols_before_filling=required_filled_cols_before_filling,
                                required_filled_cols_after_filling=required_filled_cols_after_filling,
                                drop_threshold_row_pct=drop_threshold_row_pct,
                                drop_threshold_row_quantile=drop_threshold_row_quantile,
                                drop_threshold_col_pct=drop_threshold_col_pct,
                                append_data_quality_col=append_data_quality_col)

    df_all = feature_engerneeing(dataset=df_cleaned, comp_col=comp_col, time_cols=time_cols, industry_col=industry_col, yearly_data=False)
