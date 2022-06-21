

"""
Caso de estudio - grupo 3
"""
### Librerias
import pandas as pd
from matplotlib.pyplot import figure
import seaborn as sns
import plotly.express as px
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt ### gráficos
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from pylab import rcParams
import seaborn as sb
from scipy.stats.stats import kendalltau
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

pd.options.display.max_columns = None # para ver todas las columnas
"""
SUPUESTO DE SOLUCIÓN
Crear un modelo que prediga la posible renuncia de una persona y generar un plan de acción para evitar las rotaciones en la empresa.
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

# Se utilizo el promedio para reempazar los nulos
df_general_data.describe()
df_general_data.fillna({'NumCompaniesWorked':3,'TotalWorkingYears':11,'WorkLifeBalance':3}, inplace = True) 

### en esta base de datos como los na de la razón de despidos aparecian como Na
# entonces se remplazó el na por despido 

df_retirement_info[df_retirement_info['resignationReason'].isnull()]
df_retirement_info.fillna({'resignationReason':'Fired'}, inplace = True)

### Union de los dataframe

## bases de datos a utilizar son
##df_employee_survey -------- Encuesta de satisfacción de los empleados
##df_general_data ------------ Datos de edad, departamento entre otros
##df_manager_survey--------- empleado, ambiente laboral y desempeño
#df_retirement_info---------dia de retirado, porque,y si fue renuncia o despido

## unión de las bases de datos seleccionadas por medio del ID
df = df_general_data.merge(df_employee_survey, on = 'EmployeeID', how = 'left').merge(df_manager_survey, on = 'EmployeeID', how = 'left').merge(df_retirement_info, on = 'EmployeeID', how = 'left')
df.fillna({'retirementType':'working','resignationReason':'working','Attrition':'No'}, inplace = True) 

#Características de las variables numéricas que componen la base de datos
df.describe()
 
df.dtypes # para obtener únicamente el tipo de las variables

## Eliminamos la comluna de mayores de 18 años (Over18) porque todos son >18
## Eliminar la comluna ya que tenia un número 1 y lo consideramos no importante
## Eliminar StandardHours porque todos trabajan 8 horas

df.drop(['EmployeeCount','StandardHours','Over18'], axis = 1, inplace = True) 

#mapa de calor de la correlación
figure(figsize=(20,15), dpi=80);
sns.heatmap(df.corr(), annot = True); 

#histogramas y graficos de disperción
scatter_matrix(df, figsize=(40, 35)) 
df.hist(figsize=(40, 35))

#boxplot
##comparamos el tipo de retiro con variables que con
##consideramos por "sentido común" fuertes para esos retiros
df.boxplot('PercentSalaryHike','retirementType',figsize=(5,5)) #porcetaje de aumentos de salario
df.boxplot('TotalWorkingYears','retirementType',figsize=(5,5)) #años trabajando en total, se deberia normalizar para tratar datos atipicos
df.boxplot('JobSatisfaction','retirementType',figsize=(5,5)) #satisfacción en el trabajo
df.boxplot('EnvironmentSatisfaction','retirementType',figsize=(5,5))#Satisfación con el ambiente
df.boxplot('YearsAtCompany','retirementType',figsize=(5,5)) #Años trabajando en la compañia

### EXPLORACION DE DATOS

df.columns

df = df[['EmployeeID', 'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education',
       'EducationField', 'Gender', 'JobLevel', 'JobRole',
       'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',
       'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',
       'YearsWithCurrManager', 'EnvironmentSatisfaction', 'JobSatisfaction',
       'WorkLifeBalance', 'JobInvolvement', 'PerformanceRating', 'retirementDate',
        'retirementType', 'resignationReason', 'Attrition']]


df.loc[df.Attrition=='Yes']

df['YearsAtCompany'] = df['YearsAtCompany'].astype('category')
df['Attrition'] = df['Attrition'].astype('category')

## Revisamos las personas que se retiraron cuantos años llevaban trabajando en la compañía

in_yes = df['Attrition'] == 'Yes'
in_yes.head()
att_yes = df[in_yes]
att_yes

base = att_yes.groupby(['YearsAtCompany'])[['Attrition']].count().reset_index()
base
fig = px.bar(base, x = 'YearsAtCompany', y='Attrition', color = 'Attrition', barmode = 'group', title= '<b>despedidos vs años en la compañía<b>')
# agregar detalles a la gráfica
fig.update_layout(
    xaxis_title = 'Años en la compañia',
    yaxis_title = 'Attrition',
    template = 'simple_white',
    title_x = 0.5)

fig.show()

##Mostrar el número de empleados que se fueron y se quedaron por edad
fig_dims = (12, 4)
fig, ax = plt.subplots(figsize=fig_dims)
#ax = axis
sns.countplot(x='Age', hue='Attrition', data = df, palette="colorblind", ax = ax,  edgecolor=sns.color_palette("dark", n_colors = 1));

####Ajustar un modelo para ver importancia de variables categóricas

####Crear variables para entrenar modelo
###---------------------------------------------------
#Comente esta parte ya que se utiliza la de cristhian
"""
df.info()
y=df.Attrition.replace({'Yes':'1','No':'0'})

X_dummy=pd.get_dummies(df,columns=['BusinessTravel','Department','Education','EducationField','Gender','JobLevel','JobRole','MaritalStatus','StockOptionLevel','retirementType','resignationReason'])
X_dummy.info()

X_dummy.drop(['Attrition','retirementDate'],axis = 1, inplace = True)


#imprime todos los string, su tipo de dato y valore unicos 
for column in df.columns:
    if df[column].dtype == "string":
        print(str(column) + ' : ' + str(df[column].unique()))
        print(df[column].value_counts())
        print("_________________________________________________________________")
        

## tabla de tipo de retiro 
pd.crosstab(index=df_retirement_info['retirementType'], columns=' count ')
# tabla conteo razón de la renuncia 
pd.crosstab(index=df_retirement_info['resignationReason'], columns=' count ')
#tabla conteo de la razón de la renuncia con el departamento o área de la empres
pd.crosstab(index=df['Department'], columns=df['resignationReason'], margins=True)

#tabla porcentaje de los que renunciaron por departamento
pd.crosstab(index=df['Department'], columns=df.loc[df['resignationReason']!='Fired','Attrition'], margins=True, normalize='index')
"""
#--------------------------------------------------------------
#features selection prueba 2

#Eliminar columnas que no tienen sentido para predecir "Attrition"
df = df.drop(['EmployeeID', 'resignationReason','retirementType','retirementDate'], axis=1)
df

# dummies
department_types = ('Sales', 'Research & Development', 'Human Resources')
dum_df = pd.get_dummies(df, columns=["Department"], prefix=["department_is"] )

BusinessTravel_types = ('Travel_Rarely', 'Travel_Frequently', 'Non-Travel')
dum_df = pd.get_dummies(dum_df, columns=["BusinessTravel"], prefix=["businesstravel_is"] )

EducationField_types = ('Life Sciences',            'Other',          'Medical',
        'Marketing', 'Technical Degree',  'Human Resources')
dum_df = pd.get_dummies(dum_df, columns=["EducationField"], prefix=["educationfield_is"] )

JobRole_types = ('Healthcare Representative',        'Research Scientist',
           'Sales Executive',           'Human Resources',
         'Research Director',     'Laboratory Technician',
    'Manufacturing Director',      'Sales Representative',
                   'Manager')
dum_df = pd.get_dummies(dum_df, columns=["JobRole"], prefix=["jobrole_is"] )

MaritalStatus_types = ('Married', 'Single', 'Divorced')
dum_df = pd.get_dummies(dum_df, columns=["MaritalStatus"], prefix=["maritalstatus_is"] )

#convertir a codigo genero

labelencoder = LabelEncoder()
dum_df['Gender_code'] = labelencoder.fit_transform(dum_df['Gender']) #female=0, male=1
dum_df = dum_df.drop(['Gender'], axis=1)
dum_df['Attrition_code'] = labelencoder.fit_transform(dum_df['Attrition']) #female=0, male=1
dum_df = dum_df.drop(['Attrition'], axis=1)

dum_df2 = dum_df

#separar variables de entreda y variable salida
X = dum_df2.drop('Attrition_code',axis=1)
y = dum_df2['Attrition_code']

#valores chi2
chi_scores = chi2(X,y)
chi_scores
 #graficar valores p
p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)    
p_values.plot.bar()

# Convert to categorical data by converting data to integers
X = X.astype(int)

#Seleccionar los mejores 10 de chi2
select = SelectKBest(score_func=chi2)
z = select.fit_transform(X,y)

print("After selecting best 10 features:", z.shape) 
filter = select.get_support()
features = np.array(X.columns)
 
print("All features:")
print(features)
 
print("Selected best 10:")
print(features[filter])
print(z)   

#matriz de correlación de kendall
corr = dum_df2.corr(method='kendall')
rcParams['figure.figsize'] = 30,25
sb.heatmap(corr, 
           xticklabels=corr.columns.values, 
           yticklabels=corr.columns.values, 
           cmap="YlGnBu",
          annot=True, fmt='.0%')


'''
Resultado de principales variables a elegir

correlación de kendall:
Edad
Años trabajando
Años en la compañia
Años con el mismo jefe
Estado marital
Frecuencia de viaje

Mejores 10 de chi2:
Edad
Ingreso mensual
Años trabajando
Años en la compañia
Años con el mismo jefe
Departamento
Frecuencia de viaje 
Campo de educación
Estado marital

Chi2 grafica primeras 15:
Ingreso mensual
Años trabajando
Años en la compañia
Años con el mismo jefe
Edad
Estado marital
Frecuencia de viaje
Campo de educación
Departamento
Años desde último acenso
Satisfacción del trabajo
Satisfacción del ambiente'''
dum_df2.columns
df = dum_df2[['Age', 'MonthlyIncome',       
       'TotalWorkingYears', 'YearsAtCompany',
       'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'EnvironmentSatisfaction', 'JobSatisfaction',
       'department_is_Human Resources',
       'department_is_Research & Development', 'department_is_Sales',
       'businesstravel_is_Non-Travel', 'businesstravel_is_Travel_Frequently',
       'businesstravel_is_Travel_Rarely', 'educationfield_is_Human Resources',
       'educationfield_is_Life Sciences', 'educationfield_is_Marketing',
       'educationfield_is_Medical', 'educationfield_is_Other',
       'educationfield_is_Technical Degree',
       'maritalstatus_is_Divorced',
       'maritalstatus_is_Married', 'maritalstatus_is_Single',
       'Attrition_code']]
#------------------------------------------------------------

z = df.drop('Attrition_code',axis=1)
y = df['Attrition_code']

#Modelo random forest

test_size = 0.33
seed = 3
X_train, X_test, Y_train, Y_test = train_test_split(z, y, test_size=test_size,random_state=seed)
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)

#muestra el accuracy de modelo
forest.score(X_train, Y_train)

#Matriz de confunción para los datos de prueba
cm = confusion_matrix(Y_test, forest.predict(X_test))
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
print(cm)
print('Model Testing Accuracy = "{}!"'.format(  (TP + TN) / (TP + TN + FN + FP)))

#Prueba
resultado1 = forest.score(X_test, Y_test)
print("Precisión: ",resultado1*100)

#k-fold Cross-Validation
kfold = KFold(n_splits=5, random_state=5, shuffle=True)
resultado2 = cross_val_score(forest, z, y, cv=kfold)
print("Precisión: ",resultado2.mean()*100)

#Entrenamiento aleatorio repetido
splits = 5
kfold = ShuffleSplit(n_splits=splits, test_size=test_size, random_state=seed)
resultado4 = cross_val_score(forest, z, y, cv=kfold)
print("Precisión: ",resultado4.mean()*100)

#Entrenamiento y pruebas
loo = LeaveOneOut()
resultado3 = cross_val_score(forest, z, y, cv=loo)
print("Precisión: ",resultado3.mean()*100)

# tabla de resultados de entrenamientos
tabla = pd.DataFrame()
nombre = ['Train and Test Sets','k-fold Cross-Validation','Leave One Out Cross-Validation','Repeated Random Test-Train Splits']
#Precision = [resultado1,resultado2.mean(),resultado3.mean(),resultado4.mean()]
Precision = [resultado1,resultado2.mean(),resultado3.mean(),resultado4.mean()]
tabla['Evaluador de desempeño'] = nombre
tabla['Precision'] = Precision
print(tabla)


#revisar bien estas métricas
#MÉTRICAS DE PRECISIÓN
#Porcentaje de exactitud 
kfold = KFold(n_splits=5, random_state=7, shuffle=True)
score = 'accuracy'
resultado = cross_val_score(forest,z,y,cv=kfold,scoring=score)
print("Accuracy: ",resultado.mean()*100)

#Resivir información
nuevo_dato = 'df de nuevo(s) empleado'

#transformación del nuevo dato



## cuales empleados podrian renunciar


#como dimensiono el area de recursos humanos en función de las otras áreas
##mirar la rotación de como estan en las otras 2 áreas
#cruzar con eldepartamento


#flitrar los años trabajados en la compañía por depa


#regresión con ventas e investigación y desarrolo y para recurso humanos hacer un descriptivo.
#con modelos predictivo logrmos reducir no ayuda en la rotación en RRHH
