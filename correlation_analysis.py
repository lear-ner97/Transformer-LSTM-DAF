# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 00:01:04 2024

@author: umroot
"""


#this correlation analysis block should be called at line 128 of main.py 
#######################################################################################################
features_t=['day_of_week', 'cosine_transform_day_of_year','cosine_transform_month',
            'cosine_transform_hour_of_day', 'is_weekend','is_holiday','airTemperature','windSpeed']
#features_t=['is_weekend','is_holiday','cosine_transform_hour_of_day','airTemperature','windSpeed']

lagged_features=[f"airTemperature_lag{i}" for i in range(src_lookback,0,-1)]+[f"windSpeed_lag{i}" for i in range(src_lookback,0,-1)]


src_columns=features_t+lagged_features+[f"{src_building}_lag{i}" for i in range(src_lookback+src_horizon-1,0,-1)]+[src_building]#features_t+lagged_features+
tgt_columns=features_t+lagged_features+[f"{tgt_building}_lag{i}" for i in range(tgt_lookback+tgt_horizon-1,0,-1)]+[tgt_building]#features_t+lagged_features+

src_shifted_df=src_shifted_df[src_columns]
tgt_shifted_df=tgt_shifted_df[tgt_columns]
# X_src=src_shifted_df[src_columns]
# X_tgt=tgt_shifted_df[tgt_columns]
# Compute correlations
correlations = src_shifted_df.corr()

# Select correlations between features and target vector:
  #1-compute the pearson correlation between the features and each future step
#2-apply absolute value on the correlation values
#3-compute the mean correlation of the feature with all future steps
#4-choose the features with a mean absolute correlation > 0.2

correlations_with_target = abs(correlations.iloc[:-src_horizon, -src_horizon:]).mean(axis=1)

# Print or use the correlations
#print("Correlation coefficients with target vector:")
#print(correlations_with_target)


selected_features = correlations_with_target[correlations_with_target > 0.2].index.tolist()#
#after correlation analysis we selected the complete historical load + weekend + cosine_transf_hour
################################################################################################