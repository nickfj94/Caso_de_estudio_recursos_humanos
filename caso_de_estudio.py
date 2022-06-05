"""
Caso de estudio
"""
### Librerias
import pandas as pd
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

