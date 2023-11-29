# Importaciones necesarias:
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
import os
import warnings
import pickle
warnings.filterwarnings('ignore')

# Leer el dataframe procesado
df  = pd.read_csv('../data/processed/processed.csv')

# Separamos los datos en test y train
X = df.drop(columns='Adaptivity Level')            
y = df['Adaptivity Level']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)

# Junto X e y para guardarlos en un dataframe
df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

# Los guardo en sus respectivas carpetas
df_train.to_csv('../data/train/train.csv', index=False)
df_test.to_csv('../data/test/test.csv', index=False)

# Mediante SMOTE vamos a oversamplear los datos
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Entrenamiento con RandomForestClassifier
parameters = {
    'n_estimators': [20, 40, 60, 80, 100, 120, 140, 160, 180],    # cantidad de árboles del bosque.
    'criterion': ['gini', 'entropy', 'log_loss'],                 # función para medir la calidad de la división.
    'max_leaf_nodes': [80,100,120,140,160,180,200,220],           # nodos máximos del modelo.
    'bootstrap': [True, False]                                    # si es falso, se utiliza todo el conjunto de datos para construir cada árbol.
}

modelo_rfc = RandomForestClassifier()

rfc = GridSearchCV(estimator = modelo_rfc ,
                  param_grid = parameters,
                  n_jobs = -1,                 # Número de trabajos que se ejecutarán en paralelo. -1 significa utilizar todos los procesadores
                  cv = 10,
                  scoring="accuracy")

rfc.fit(X_train_resampled, y_train_resampled)
final_rfc = rfc.best_estimator_
final_rfc.fit(X_train_resampled, y_train_resampled)

filename = 'trained_model_rfc'
with open(filename, 'wb') as archivo_salida:
    pickle.dump(final_rfc, archivo_salida)

# Entrenamiento con GradientBoostingClassifier
parameters = {
    'loss': ['log_loss', 'deviance', 'exponential'],     # La función de pérdida que se va a optimizar.
    'learning_rate': [0.6,0.7,0.8,0.9,1],                # La tasa de aprendizaje reduce la contribución de cada árbol.
    'n_estimators': [20,40,60,80,100,120,140],           # El número de etapas de impulso a realizar.
    'max_depth': [8,10,12,14,16]                         # Profundidad máxima de los estimadores de regresión individuales.
}
modelo_gbc = GradientBoostingClassifier()

gbc = GridSearchCV(estimator = modelo_gbc,
                  param_grid = parameters,
                  n_jobs = -1,
                  cv = 10,
                  scoring="accuracy")

gbc.fit(X_train_resampled, y_train_resampled)
final_gbc = gbc.best_estimator_
final_gbc.fit(X_train_resampled, y_train_resampled)

filename = 'trained_model_gbc'
with open(filename, 'wb') as archivo_salida:
    pickle.dump(final_gbc, archivo_salida)

# Entrenamiento con SVC
parameters = {
    'C': [15,20,25,30],                                  # Parámetro de regularización.
    'degree': [1,2,3,4,5,6,7],                           # Grado de la función kernel polinomial.
    'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],      # Especifica el tipo de núcleo que se utilizará en el algoritmo.
    'gamma': ['scale', 'auto']                           # Coeficiente kernel.
}

modelo_svc = svm.SVC()

svc = GridSearchCV(estimator = modelo_svc ,
                  param_grid = parameters,
                  n_jobs = -1,
                  cv = 10,
                  scoring="accuracy")

svc.fit(X_train_resampled, y_train_resampled)
final_svc = svc.best_estimator_
final_svc.fit(X_train_resampled, y_train_resampled)

filename = 'trained_model_svc'
with open(filename, 'wb') as archivo_salida:
    pickle.dump(final_svc, archivo_salida)

# Entreno con BagginClassifier
parameters = {
    'estimator': [svm.SVC(), DecisionTreeClassifier()],        # El estimador base para ajustarse a subconjuntos aleatorios del conjunto de datos.
    'n_estimators': [50,60,70,80,90,100],                      # El número de estimadores de base en el conjunto.   
    'max_samples': [500,520,540,560,580],                      # El número de muestras que se extraerán de X para entrenar cada estimador base.
    'bootstrap': [True,False],                                 # Si las muestras se extraen con reemplazo.
    'max_features': [8,9,10,11,12,13,14,15]                    # El número de características que se extraerán de X para entrenar cada estimador base 
}

modelo_bc = BaggingClassifier()

bc = GridSearchCV(estimator = modelo_bc ,
                  param_grid = parameters,
                  n_jobs = -1,
                  cv = 10,
                  scoring = "accuracy")

bc.fit(X_train_resampled, y_train_resampled)
final_bc = bc.best_estimator_
final_bc.fit(X_train_resampled, y_train_resampled)

filename = 'trained_model_bc'
with open(filename, 'wb') as archivo_salida:
    pickle.dump(final_bc, archivo_salida)

# Entreno con AdaBoostClassifier
parameters = {
    'estimator': [svm.SVC(), DecisionTreeClassifier()],     # El estimador base a partir del cual se construye el conjunto potenciado. 
    'n_estimators': [20,30,40,50,60,70,80],                 # El número máximo de estimadores en los que finaliza el impulso.
    'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6]              # Peso aplicado a cada clasificador en cada iteración de impulso.
}

modelo_abc = AdaBoostClassifier()

abc = GridSearchCV(estimator = modelo_abc ,
                  param_grid = parameters,
                  n_jobs = -1,
                  cv = 10,
                  scoring="accuracy")

abc.fit(X_train_resampled, y_train_resampled)
final_abc = abc.best_estimator_
final_abc.fit(X_train_resampled, y_train_resampled)

filename = 'trained_model_abc'
with open(filename, 'wb') as archivo_salida:
    pickle.dump(final_abc, archivo_salida)

# Entrno con PCA + SVM
pipe_gs_rf = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', svm.SVC())
])
params = {
    "scaler" : [StandardScaler(), MinMaxScaler(), None],
    "pca__n_components": [5,6,7,8,9,10,11,12,13,14,15,16],
    'classifier__C': [15,20,25,30],                                   
    'classifier__degree': [1,2,3,4,5,6,7],                           
    'classifier__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],      
    'classifier__gamma': ['scale', 'auto'] 
}
gs = GridSearchCV(pipe_gs_rf, params, cv=10, scoring="accuracy", n_jobs=-1)
gs.fit(X_train_resampled, y_train_resampled)
final_gs = gs.best_estimator_
final_gs.fit(X_train_resampled, y_train_resampled)

filename = 'trained_model_gs'
with open(filename, 'wb') as archivo_salida:
    pickle.dump(final_gs, archivo_salida)