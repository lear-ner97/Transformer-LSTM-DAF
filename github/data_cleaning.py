# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:08:02 2024

@author: umroot
"""
import pandas as pd


def IQR_cleaning(df):
    df_copy = df.copy()
    for column in df_copy.select_dtypes(include=['number']).columns:#[1:]:#except the timestamp
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        median=df_copy[column].median()
        IQR = Q3 - Q1
        df_copy[column] = df_copy[column].apply(
        lambda x: median if x < Q1-1.5*IQR or x > Q3+1.5*IQR else x
    )
    return df_copy



building_data=pd.read_csv('electricity_cleaned.txt')
selected_meters=building_data[['timestamp','Robin_education_Julius',
                            'Robin_education_Billi',
                            'Moose_education_Ricardo',
                            'Robin_office_Maryann',
                            'Robin_office_Antonina']]
selected_meters.to_csv('selected_meters.csv')
selected_meters_clean=IQR_cleaning(selected_meters)
selected_meters_clean['is_weekend'] = (pd.to_datetime(selected_meters_clean['timestamp']).dt.weekday >= 5).astype(int)
selected_meters_clean.to_csv('clean_genome_meters.csv',index=False)