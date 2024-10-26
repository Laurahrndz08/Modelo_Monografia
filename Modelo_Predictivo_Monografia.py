#monografia - Modelo predictivo para la clasificacion si una persona renueva o no (Continua o no)
                                   
#======================Librerias para manipular=====================
# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gestion de librerias
# ==============================================================================
from importlib import reload

# Matemáticas y estadísticas
# ==============================================================================
import math

# Preparación de datos
# ==============================================================================
#from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import LocalOutlierFactor

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')


#===============DATASET============================
# los datos para el modelo: recueda no hemos sacado la muestra.
ruta_archivo ="C:\\Users\\lrebolledo\\Documents\\Monografia\\Base_Monografia.csv"
df  = pd.read_csv(ruta_archivo, delimiter=';')
print(df.head())  # Verificar lo que se monto
print(f"Total registros: {len(df)}")
#=====Exploracion de DATASET y sus variables===================================
print(df.info())
#Lista de variables categóricas
catCols = df.select_dtypes(include = ["object", 'category']).columns.tolist()
#Se elimina la columna estado de la lista de variables Categóricas ya que es nuestra variable de salida
catCols.remove('Estado')
print(df[catCols].head(2))
#Lista de variables numéricas
numCols=df.select_dtypes(include = ['float64','int32','int64']).columns.tolist()
#print(df[numCols].head(2))



#==========================para evaluar========================
# Distribución de cada variable categórica en el conjunto de datos

#for col in catCols:
 #  print("="*5 + f" {col} " + "="*20)
  # print(df[col].value_counts())
   #print()
#De aqui se evidencia como reorganizar como los metodos de pago. Unificarlos en 3 (PAGO EN CAJA, FACTURA{FACTURA, CONVENIO,CUENTA DE COBRO}, PSE{DEBITO,DESCUENTO DE NOMINA,TARJETA DE CREDITO})


#VARIABLE DE SALIDA
print(df.groupby('Estado').Estado.count().sort_values(ascending=False))


#Cambios en la data 

# Cambiar 'sex' de 'M' y 'F' a 1 y 0
df['Genero'] = df['Genero'].replace({'M': 1, 'F': 0,'m': 1, 'f': 0}).astype(int)
df['Tipo_Municipio'] = df['Tipo_Municipio'].replace({'Urbano': 1, 'Rural': 0}).astype(int)
df['Metodo_Pago'] = df['Metodo_Pago'].replace({'Factura': 1, 'Cuenta de cobro': 1,'Convenio': 1,'Pago en Caja': 0,'Débito Automático': 2,'Descuento de nómina': 2,'Tarjeta crédito': 2}).astype(int)

for col in catCols:
   print("="*5 + f" {col} " + "="*20)
   print(df[col].value_counts())
   print()