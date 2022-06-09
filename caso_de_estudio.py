"""
Caso de estudio - grupo 3
"""
### Librerias
import pandas as pd
import sqlite3 as sql
import numpy as np


"""
SUPUESTO DE SOLUCIÓN
Crear un modelo que prediga la posible renuncia de una persona y generar un plan de acción para diminuirlas.
"""

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

# código para que los números decimales con los cuales se va a trabajar aparezcan con dos decimales
pd.options.display.float_format = '{:.2f}'.format 

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
### estas columnas estaban nulas en las bases de los entradas y salidas
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


# Se utilizo el promedio para reempazar los nulos
df_employee_survey.describe()
df_employee_survey.fillna({'EnvironmentSatisfaction':3,'JobSatisfaction':3,'WorkLifeBalance':3}, inplace = True) 


#teniedo en cuenta la descripción del código, podemos revisar los promedios medianas y demás
#para procesar los nulos

# Se utilizo el promedio para reempazar los nulos
df_general_data.describe()
df_general_data.fillna({'NumCompaniesWorked':3,'TotalWorkingYears':11,'WorkLifeBalance':3}, inplace = True) 


### en esta base de datos como los na de la razón de despidos aparecian como Na
# entonces se remplazó el na por despido 

df_retirement_info[df_retirement_info['resignationReason'].isnull()]
df_retirement_info.fillna({'resignationReason':'Fired'}, inplace = True)

###Revisar columnas de las bases
df_employee_survey.columns
df_general_data.columns
df_manager_survey.columns
df_retirement_info.columns



df_retirement_info["resignationReason"].unique()
df_retirement_info["retirementType"].unique()



### Union de los dataframe

## bases de datos a utilizar son
##df_employee_survey -------- Encuesta de satisfacción de los empleados
##df_general_data ------------ Datos de edad, departamento entre otros
##df_manager_survey--------- empleado, ambiente laboral y desempeño

#lA SIGUIENTE BASE SERÁ UTILIZADA PERO NO LA UNIREMOS 
#df_retirement_info---------dia de retirado, porque,y si fue renuncia o despido

## unión de las bases de datos seleccionadas por medio del ID
df = df_general_data.merge(df_employee_survey, on = 'EmployeeID', how = 'left').merge(df_manager_survey, on = 'EmployeeID', how = 'left')
#.merge(df_retirement_info, on = 'EmployeeID', how = 'left')

df.head()
df.info()
df.isnull().sum()


#Revisión de los datos que contienien las siguientes columnas
df['EnvironmentSatisfaction'].unique()
df['JobSatisfaction'].unique()
df['WorkLifeBalance'].unique()
df['EducationField'].unique()
df['EmployeeCount'].unique()
df['JobLevel'].unique()
df['JobRole'].unique()
df['PercentSalaryHike'].unique()


#Características de las variables numéricas que componen la base de datos
df.describe() 

df.dtypes # para obtener únicamente el tipo de las variables

""" Revisar si es necesario convertir variables dumis
se puede usar este codigo depd.get_dummies(df['TIPO_GEOCOD']).head(3) 
# para trabajar con las variables categóricas también podemos convertirlas en variable dummies
"""

## SUPUESTOS ###
##El tiempo de entrenmiento (capacitación)puede ser importante

## Eliminar la comluna de mayores de 18 años Over18
df["Over18"].unique()
df.drop(['Over18'], axis = 1, inplace = True) # Para borrar columnas se pone axis = 1

## Eliminar la comluna ya que tenia un unmero 1 y lo consideramos no importante
df["EmployeeCount"].unique()
df.drop(['EmployeeCount'], axis = 1, inplace = True)

## Se puede convertir la variable BusinessTravel a dumi

df



 
