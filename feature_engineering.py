# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 23:51:28 2024

@author: umroot
"""
import pandas as pd
import numpy as np
import holidays

#extract weather data for Robin site
weather=pd.read_csv('data\weather.txt')
selected_columns=['timestamp','airTemperature','cloudCoverage','precipDepth1HR','precipDepth6HR','windSpeed']
selected_features=weather[weather.site_id=='Robin'].reset_index(drop=True)[selected_columns]
# Step 1: Convert timestamp_column to datetime
selected_features['timestamp'] = pd.to_datetime(selected_features['timestamp'])#, format='%y-%m-%d %H:%M:%S')
# Step 2: Extract the day of the year, month, day of week, hour of day
selected_features['day_of_year'] = selected_features['timestamp'].dt.dayofyear
selected_features['month'] = selected_features['timestamp'].dt.month  # Extract month (1-12)
selected_features['day_of_week'] = selected_features['timestamp'].dt.dayofweek  # Extract day of the week (Monday=0, Sunday=6)
selected_features['hour'] = selected_features['timestamp'].dt.hour  # Extract hour of the day (0-23)
# cosine transformation
# Normalize the calendar features to range [0, 2*pi]
#♣max_day_of_year = 365  # or 366 for leap years
# if we apply cosine transform to day of week then tuesdday will be the same as sunday
# if we don't apply cosine transform to hour of day then midnight will be different from 23h
max_month = 12
#max_day_of_week = 7
max_hour_of_day= 24

#normalized_day_of_year = (2 * np.pi * selected_features['day_of_year']) / max_day_of_year
#df_transformed = pd.concat([df_2016, df_2017])normalized_day_of_year = (2 * np.pi * selected_features['day_of_year']) / max_day_of_year
normalized_month = (2 * np.pi * selected_features['month']) / max_month
#normalized_day_of_week = (2 * np.pi * selected_features['day_of_week']) / max_day_of_week
normalized_hour_of_day = (2 * np.pi * selected_features['hour']) / max_hour_of_day

#applying cosine transform on day of year helps to capture yearly seasonality
# if we use day of year directly then day 1 (1st of january) will be different from december 31 (365)
# Apply cosine transformation for 2016 (max_day_of_year=366)
selected_features_2016 = selected_features[selected_features['timestamp'].dt.year == 2016].copy()
max_day_of_year_2016 = 366
selected_features_2016['normalized_day_of_year'] = (2 * np.pi * selected_features_2016['day_of_year']) / max_day_of_year_2016
selected_features_2016['cosine_transform'] = np.cos(selected_features_2016['normalized_day_of_year'])

# Apply cosine transformation for 2017 (max_day_of_year=365)
selected_features_2017 = selected_features[selected_features['timestamp'].dt.year == 2017].copy()
max_day_of_year_2017 = 365
selected_features_2017['normalized_day_of_year'] = (2 * np.pi * selected_features_2017['day_of_year']) / max_day_of_year_2017
selected_features_2017['cosine_transform'] = np.cos(selected_features_2017['normalized_day_of_year'])

# Apply cosine transformation
selected_features['cosine_transform_day_of_year'] = pd.concat([selected_features_2016,selected_features_2017])['cosine_transform']
selected_features['cosine_transform_month'] = np.cos(normalized_month)
#day of week not considered because if we apply cosine fct then sunday=tuesday
#selected_features['cosine_transform_day_of_week'] = np.cos(normalized_day_of_week)
selected_features['cosine_transform_hour_of_day'] = np.cos(normalized_hour_of_day)

# Add column indicating weekend (1 for weekend, 0 for weekday)
selected_features['is_weekend'] = (selected_features['day_of_week'] >= 5).astype(int)

# Define holidays for England
england_holidays = holidays.CountryHoliday('GB', prov='ENG')
# Add column indicating holiday (1 for holiday, 0 for non-holiday)
selected_features['is_holiday'] = selected_features['timestamp'].apply(lambda x: 1 if x in england_holidays else 0)

#drop day of year, month, hour of day
selected_features=selected_features.drop(['day_of_year','month','hour'], axis=1)
selected_features.to_csv('robin_weather_calendar.csv',sep=',',index=False)