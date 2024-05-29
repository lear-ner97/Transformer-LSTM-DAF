# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:11:08 2024

@author: umroot
"""

import matplotlib.pyplot as plt
import pandas as pd


# 1- to visualize The Building Data Genome Project 2 
selected_meters=pd.read_csv('clean_genome_meters.csv')

#choose building name from the following list
# Moose_education_Ricardo , Robin_education_Billi , Robin_office_Maryann, Robin_office_Antonina 

building_name='Moose_education_Ricardo' #fill the name here
data=selected_meters[['timestamp',building_name]]
data['timestamp'] = pd.to_datetime(data['timestamp'])
plt.plot(data['timestamp'], data[building_name])
plt.xticks(rotation=45, ha='right')
plt.xlabel('time')
plt.ylabel('load')
plt.show()

# 2- to visualize the Malaysian electricity consumption data
data = pd.read_csv('malaysia_all_data_for_paper.csv',sep=';')
data['time'] = pd.to_datetime(data['time'])
plt.plot(data['time'], data['load'])
plt.xticks(rotation=45, ha='right')
plt.xlabel('time')
plt.ylabel('load (MW)')
plt.show()