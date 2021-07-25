
def get_dataset_registry():
    dataset_registry = {'handpicked_dataset': {'sql_table': 'entire_small_dataset2_unique', 'iter_cols': ['data_year', 'data_qrt'], 'company_col': 'ric', 'y_col': ['y_roe'], 'industry_col': 'industry', 'category_cols': [], 'date_cols': []}}

    return dataset_registry