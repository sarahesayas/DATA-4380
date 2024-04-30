# data_cleaning.py

import pandas as pd
from scipy import stats

def data_types(df):
    # Dates
    df["fecha_alta"]=pd.to_datetime(df["fecha_alta"])
    df["ult_fec_cli_1t"]=pd.to_datetime(df["ult_fec_cli_1t"])

    # Numeric
    df.loc[df['antiguedad']=='     NA','antiguedad']=None
    df["antiguedad"]=pd.to_numeric(df["antiguedad"])
    df.loc[df['age']==' NA','age']=None
    df["age"]=pd.to_numeric(df["age"])

    return df


def missing_values(df):
    missing = ['ind_empleado', 'pais_residencia', 'sexo',
           'age', 'fecha_alta', 'ind_nuevo', 'antiguedad', 'indrel',
           'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext',
           'canal_entrada', 'indfall', 'cod_prov',
           'nomprov', 'ind_actividad_cliente', 'segmento', 'ind_nomina_ult1','ind_nom_pens_ult1']

    df.dropna(subset=missing)

    return df


def remove_id_column(df):
    df.drop(labels=['ncodpers', 'fecha_dato', 'fecha_alta'], axis=1, inplace=True)
    return df


def invalid_entries(df):
    df.drop_duplicates()

    df.drop('indrel_1mes', axis=1, inplace=True)

    df.loc[df['age']<18, 'age'] = 18

    return df


def handle_outliers(df):
    for col in ['ind_nuevo', 'indrel', 'conyuemp', 'tipodom', 'cod_prov']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)

        df.loc[df[col] > upper_bound, col] = Q3
        df.loc[df[col] < lower_bound, col] = Q1

    return df
