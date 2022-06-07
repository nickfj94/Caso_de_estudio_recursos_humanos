"""
Caso de estudio - grupo 3
"""
### Librerias
import pandas as pd
import sqlite3 as sql
import numpy as np


### Carga archivos

employee_survey = 'https://raw.githubusercontent.com/nickfj94/Caso_de_estudio_recursos_humanos/main/employee_survey_data.csv'
general_data = 'https://raw.githubusercontent.com/nickfj94/Caso_de_estudio_recursos_humanos/main/general_data.csv'
in_time = 'https://raw.githubusercontent.com/nickfj94/Caso_de_estudio_recursos_humanos/main/in_time.csv'
manager_survey = 'https://raw.githubusercontent.com/nickfj94/Caso_de_estudio_recursos_humanos/main/manager_survey_data.csv'
out_time = 'https://raw.githubusercontent.com/nickfj94/Caso_de_estudio_recursos_humanos/main/out_time.csv'
retirement_info = 'https://raw.githubusercontent.com/nickfj94/Caso_de_estudio_recursos_humanos/main/retirement_info.csv'

df_employee_survey = pd.read_csv(employee_survey)
df_general_data = pd.read_csv(general_data, sep = ';') #Separado por ;
df_in_time = pd.read_csv(in_time)
df_manager_survey = pd.read_csv(manager_survey)
df_out_time = pd.read_csv(out_time)
df_retirement_info = pd.read_csv(retirement_info, sep = ';') #Separado por ;

### Resumen con información tablas faltantes y tipos de variables y hacer correcciones

df_employee_survey.info()
df_general_data.info()
df_in_time.info()
df_manager_survey.info()
df_out_time.info()
df_retirement_info.info()

### Convertir los datos

df_employee_survey = df_employee_survey.convert_dtypes()
df_general_data = df_general_data.convert_dtypes()
df_in_time = df_in_time.convert_dtypes()
df_manager_survey = df_manager_survey.convert_dtypes()
df_out_time = df_out_time.convert_dtypes()
df_retirement_info = df_retirement_info.convert_dtypes()

### Convertir campos a formato fecha
"""
Falta los de df_in_time y df_out_time
"""

df_retirement_info['retirementDate'] = pd.to_datetime(df_retirement_info['retirementDate'])


### Eliminar columnas

df_in_time = df_in_time.drop('Unnamed: 0', axis = 1)
df_out_time = df_out_time.drop('Unnamed: 0', axis = 1)

### Imprimir los primeros valores de cada dataframe

df_employee_survey.head()
df_general_data.head()
df_in_time.head()
df_manager_survey.head()
df_out_time.head()
df_retirement_info.head()

### Tratamiento nulos

df_employee_survey[df_employee_survey['EnvironmentSatisfaction'].isnull()]
df_employee_survey[df_employee_survey['JobSatisfaction'].isnull()]
df_employee_survey[df_employee_survey['WorkLifeBalance'].isnull()]

df_employee_survey.dropna(inplace=True)

df_general_data[df_general_data['NumCompaniesWorked'].isnull()]
df_general_data[df_general_data['TotalWorkingYears'].isnull()]

df_general_data.dropna(inplace = True)

df_retirement_info[df_retirement_info['resignationReason'].isnull()]
df_retirement_info.fillna({'resignationReason':'Fired'}, inplace = True)

###Revisar columnas de las bases
df_employee_survey.columns
df_general_data.columns
df_manager_survey.columns
df_retirement_info.columns


## 
df_retirement_info["resignationReason"].unique()
df_retirement_info["retirementType"].unique()

"""
supuestos de solución

aplicación de aplicativo suponiendo que ya pasaron los criterios de selección
luego los que quedan através de un modelo supervisado probablemente de regresión
tratará de predecir el tiempo de duración en la empresa y ya con el mismo aplicativo se elige
los que tengan mayor puntaje

"""

### Union de los dataframe

df = df_general_data.merge(df_employee_survey, on = 'EmployeeID', how = 'left').merge(df_manager_survey, on = 'EmployeeID', how = 'left').merge(df_retirement_info, on = 'EmployeeID', how = 'left')

df.head()
df.info()

