import os, sys, zlib, hashlib, warnings, math, datetime, random
import psycopg2, psycopg2.extras, pickle, progressbar, time, hashlib
import pandas as pd
import numpy as np
import tensorflow as tf

### Add other shared functions ###
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import z_helpers as my
##################################
from a_data_cleaning import get_clean_data


def get_pct_change(df, time_cols=['period_year', 'period_qrt'], company_cols=['gvkey'], reset_index=True):
    df_cp = df.copy()
    df_cp = df_cp.set_index(company_cols + time_cols)
    df_cp = df_cp.sort_index()
    df_cp = df_cp.select_dtypes(np.number).groupby(level=company_cols).pct_change()
    if reset_index:
        df_cp = df_cp.reset_index()
    return df_cp

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


def compare_against_industry(df_comps, df_industry, time_cols, industry_col, as_int=False):
    comb = df_comps.merge(df_industry, on=time_cols + [industry_col], how="left", suffixes=("_comp", "_ind"))
    cols = df_comps.columns.tolist()
    cols = [i for i in cols if i in df_industry.columns.tolist() and i not in time_cols and i != industry_col]
    new_df = comb[time_cols + [industry_col]].copy()
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
    new_df['y_eps'] = df_abs['eps']
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

    df_ind_avg = industry_average(new_df, time_cols=time_cols, industry_cols=[industry_col], reset_index=True)

    dummy_comp_vs_ind = compare_against_industry(df_comps=new_df, df_industry=df_ind_avg, time_cols=time_cols, industry_col=industry_col, as_int=True)

    return dummy_comp_vs_ind



if __name__ == '__main__':
    data_version = 'handpicked_dataset'
    recache = False
    comp_col = 'ric'
    time_cols = ['data_year', 'data_qrt']
    industry_col = 'industry'

    df = get_clean_data(data_version=data_version, recache=recache, comp_col=comp_col, time_cols=time_cols, industry_col=industry_col)

    df_ind_avg = industry_average(df, time_cols=time_cols, industry_cols=[industry_col], reset_index=True)
    df = df.merge(df_ind_avg[time_cols + [industry_col] + ['capitalexpenditures', 'randd']], how='left', on=time_cols + [industry_col], validate='many_to_one', suffixes=['', '_avg'])
    df_shift = get_shifted_df(df, shift=1, time_cols=time_cols, company_cols=[comp_col], reset_index=True).fillna(method='bfill')
    df = df.merge(df_shift[time_cols + [comp_col] + ['sales', 'employeenum', 'dividendperstock']], how='left', on=time_cols + [comp_col], validate='many_to_one', suffixes=['', '_sft_1'])

    df_pct = get_pct_change(df, time_cols=time_cols, company_cols=[comp_col], reset_index=True)
    df_pct_shift4 = get_shifted_df(df_pct, shift=4, time_cols=time_cols, company_cols=[comp_col], reset_index=True).fillna(method='bfill')
    df_pct = df.merge(df_pct_shift4[time_cols + [comp_col] + ['capextoassets']], how='left', on=time_cols + [comp_col], validate='many_to_one', suffixes=['', '_sft_4'])

    df_lev_thi = lev_thiagaranjan_signs(df_abs=df, df_pct=df_pct, iter_col=time_cols, company_col=comp_col)
    df_ou_penn = ou_pennmann_signs(df_abs=df, df_pct=df_pct, iter_col=time_cols, company_col=comp_col)
    df_xue_zha = xue_zhang_signs(df_abs=df, df_pct=df_pct, iter_col=time_cols, company_col=comp_col, industry_col=industry_col)

    print(df_lev_thi, df_ou_penn, df_xue_zha)