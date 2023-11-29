# Importaciones necesarias:
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Datos originales:
df = pd.read_csv('../data/raw/students_adaptability.csv')

# Mapeo:
mapeo_age = {'1-5': 0, '6-10': 1, '11-15': 2, '16-20': 3, '21-25': 4, '26-30': 5}
df['Age'] = df['Age'].map(mapeo_age)
mapeo_edu_level = {'School': 0, 'College': 1, 'University': 2}
df['Education Level'] = df['Education Level'].map(mapeo_edu_level)
mapeo_it = {'No': 0, 'Yes': 1}
df['IT Student'] = df['IT Student'].map(mapeo_it)
mapeo_location = {'No': 0, 'Yes': 1}
df['Location'] = df['Location'].map(mapeo_location)
mapeo_Load_shedding = {'Low': 0, 'High': 1}
df['Load-shedding'] = df['Load-shedding'].map(mapeo_Load_shedding)
mapeo_fin_cond = {'Poor': 0, 'Mid': 1, 'Rich': 2}
df['Financial Condition'] = df['Financial Condition'].map(mapeo_fin_cond)
mapeo_int_type = {'Mobile Data': 0, 'Wifi': 1}
df['Internet Type'] = df['Internet Type'].map(mapeo_int_type)
mapeo_net_type = {'2G': 0, '3G': 1, '4G': 2}
df['Network Type'] = df['Network Type'].map(mapeo_net_type)
mapeo_class = {'0': 0, '1-3': 1, '3-6': 2}
df['Class Duration'] = df['Class Duration'].map(mapeo_class)
mapeo_lms = {'No': 0, 'Yes': 1}
df['Self Lms'] = df['Self Lms'].map(mapeo_lms)
mapeo_Device = {'Mobile': 0, 'Tab': 1, 'Computer': 2}
df['Device'] = df['Device'].map(mapeo_Device)
mapeo_adap = {'Low': 0, 'Moderate': 1, 'High': 2}
df['Adaptivity Level'] = df['Adaptivity Level'].map(mapeo_adap)
# Dummies:
df_dummies = pd.get_dummies(df['Institution Type'], prefix='Institution_Type')
df_dummies = df_dummies.astype(int)
df = pd.concat([df, df_dummies], axis=1)
df_dummies = pd.get_dummies(df['Gender'], prefix='Gender')
df_dummies = df_dummies.astype(int)
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(columns=['Institution Type', 'Gender'])

# Los guardo en sus respectivas carpetas
df.to_csv('../data/processed/processed.csv', index=False)
