# Importaciones necesarias:
import pandas as pd
import pickle


# Cargar dataset test
df_test  = pd.read_csv("../data/test/test.csv")
X_test = df_test.iloc[:, :-1]
y_test = df_test['Adaptivity Level']

# Cargar modelos
with open('../models/final_model_svc', 'rb') as archivo_entrada:
        final_svc = pickle.load(archivo_entrada)
with open('../models/trained_model_abc', 'rb') as archivo_entrada:
        final_abc = pickle.load(archivo_entrada)
with open('../models/trained_model_bc', 'rb') as archivo_entrada:
        final_bc = pickle.load(archivo_entrada)
with open('../models/trained_model_gbc', 'rb') as archivo_entrada:
        final_gbc = pickle.load(archivo_entrada)
with open('../models/trained_model_gs', 'rb') as archivo_entrada:
        final_gs = pickle.load(archivo_entrada)
with open('../models/trained_model_rfc', 'rb') as archivo_entrada:
        final_rfc = pickle.load(archivo_entrada)

# Evaluo modelos
y_pred = final_rfc.predict(X_test)
print(final_rfc.score(X_test,y_test))
y_pred = final_gs.predict(X_test)
print(final_gs.score(X_test,y_test))
y_pred = final_gbc.predict(X_test)
print(final_gbc.score(X_test,y_test))
y_pred = final_bc.predict(X_test)
print(final_bc.score(X_test,y_test))
y_pred = final_abc.predict(X_test)
print(final_abc.score(X_test,y_test))
y_pred = final_svc.predict(X_test)
print(final_svc.score(X_test,y_test))
