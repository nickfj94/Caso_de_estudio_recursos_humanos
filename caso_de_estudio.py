

"""
Caso de estudio - grupo 3
"""
### Librerias
import pandas as pd
from matplotlib.pyplot import figure
import seaborn as sns
import plotly.express as px
from pandas.plotting import scatter_matrix
from sklearn import tree ###para ajustar arboles de decisión
from sklearn.tree import export_text ## para exportar reglas del árbol
import matplotlib.pyplot as plt ### gráficos
pd.options.display.max_columns = None # para ver todas las columnas

from sklearn import feature_selection

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

## SUPUESTOS ###
##El tiempo de entrenamiento (capacitación)puede ser importante

## Eliminamos la comluna de mayores de 18 años (Over18) porque todos son >18
df["Over18"].unique()
df.drop(['Over18'], axis = 1, inplace = True) # Para borrar columnas se pone axis = 1


## Eliminar la comluna ya que tenia un número 1 y lo consideramos no importante
## Eliminar StandardHours porque todos trabajan 8 horas


df.drop(['EmployeeCount','StandardHours'], axis = 1, inplace = True) # ,'Attrition','retirementDate'


## se cambian los dtype de acuerdo a si se debe tratar como cadena de texto o número
df['Education'] = df['Education'].astype('string')
df['EmployeeID'] = df['EmployeeID'].astype('string')
df['JobLevel'] = df['JobLevel'].astype('string')
df['StockOptionLevel'] = df['StockOptionLevel'].astype('string')
df['resignationReason'] = df['resignationReason'].astype('string')
df['retirementType'] = df['retirementType'].astype('string')
df['retirementDate'] = pd.to_datetime(df['retirementDate'])

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


####Ajustar un modelo para ver importancia de variables categóricas

####Crear variables para entrenar modelo
df.info()
y=df.Attrition.replace({'Yes':'1','No':'0'})

X_dummy=pd.get_dummies(df,columns=['BusinessTravel','Department','Education','EducationField','Gender','JobLevel','JobRole','MaritalStatus','StockOptionLevel','retirementType','resignationReason'])
X_dummy.info()

X_dummy.drop(['Attrition','retirementDate'],axis = 1, inplace = True)

#entrenar modelo
rtree=tree.DecisionTreeRegressor(max_depth=5)
rtree=rtree.fit(X=X_dummy,y=y)

####Analizar resultados del modelo
r = export_text(rtree,feature_names=X_dummy.columns.tolist(),show_weights=True)
print(r)
plt.figure(figsize=(40,40))
tree.plot_tree(rtree,fontsize=9,impurity=False,filled=True)
plt.show()

#####HAcer lista de variables importantes
d={"columna":X_dummy.columns,"importancia": rtree.feature_importances_}
df_import=pd.DataFrame(d)
pd.set_option('display.max_rows', 100)
df_import.sort_values(by=['importancia'],ascending=0)


pd.crosstab(index=df_retirement_info['retirementType'], columns=' count ')
pd.crosstab(index=df_retirement_info['resignationReason'], columns=' count ')

pd.crosstab(index=df['Department'], columns=df['resignationReason'], margins=True)


pd.crosstab(index=df['Department'], columns=df.loc[df['resignationReason']!='Fired','Attrition'], margins=True, normalize='index')

pd.crosstab(index=df['YearsAtCompany'], columns=df.loc[df['resignationReason']!='Fired','Attrition'], margins=True, normalize='index')



##FEATURE SELECTION
X_dummy

X_dummy
y=df.Attrition.replace({'Yes':'1','No':'0'})
y=y.astype(int)


feature_selection.f_classif(X_dummy, y)

feature_selection.chi2(X_dummy, y)

#como dimensiono el area de recursos humanos en función de las otras áreas
##mirar la rotación de como estan en las otras 2 áreas
#cruzar con eldepartamento


#flitrar los años trabajados en la compañía por depa


#regresión con ventas e investigación y desarrolo y para recurso humanos hacer un descriptivo.
#con modelos predictivo logrmos reducir no ayuda en la rotación en RRHH


