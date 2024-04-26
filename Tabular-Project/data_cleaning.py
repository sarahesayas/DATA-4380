# data_cleaning.py

import pandas as pd
from scipy import stats


def filter_nonzero_columns(df):
    filtered_df = df.loc[:, (df != 0).any(axis=0)]
    print("Original shape: {}".format(df.shape))
    print("Filtered shape: {}".format(filtered_df.shape))
    return filtered_df



def remove_id_column(df):
    df.drop(labels=['ID'], axis=1, inplace=True)
    return df


def remove_outliers_zscore(df, threshold=5):
    numeric_df = df.select_dtypes(include=['number'])
    
    z_scores = stats.zscore(numeric_df)
    
    outliers = (z_scores > threshold).any(axis=1)
    
    
    cleaned_df = df[~outliers]
    
    return cleaned_df
