import streamlit as st
import pandas as pd
from PIL import Image
import streamlit.components.v1 as c
import pickle

st.set_page_config(page_title="Adaptabilidad",
                   page_icon=":electric_plug:")

color_de_fondo = "#363636"

seleccion = st.sidebar.selectbox("Selecciona menu", ['Home','Datos', 'Oversampling', "Modelos de clasificación", 'Predicción'])


if seleccion == "Home":
    st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Adaptabilidad a la educación online</h1>", unsafe_allow_html=True)
    img = Image.open("../images/alumno.jpg")
    imagen = img.resize((800, 400))
    st.image(imagen)
    texto = """
                El siguiente proyecto tiene como objetivo predecir cómo se adaptarán los alumnos a las clases online en base a una serie de atributos.
                Para ello se extrajeron los datos de [Kaggle](https://www.kaggle.com/datasets/mdmahmudulhasansuzan/students-adaptability-level-in-online-education).
                La variable a predecir, a la que llamaremos target, está compuesta por las siguientes categorías:

                Los valores de target son:
                - Low: adaptabilidad baja.
                - Moderate: adaptabilidad moderada.
                - High: adaptabilidad alta.

                Las variables con las que contamos para hacer las predicciones son:
                - Gender: género del estudiante.
                - Age: rango de edad.
                - Education Level: nivel de la Institución Educativa.
                - Institution Type: si la Institución es o no gubernamental.
                - IT Student: si es o no un alumno IT.
                - Location: si el alumno vive en la ciudad o no.
                - Load-shedding: nivel de deslastre de carga.
                - Financial Condition: condición financiera de la familia del estudiante.
                - Internet Type: tipo de conexión a internet.
                - Network Type: tipo de red.
                - Class Duration: duración diaria de las clases.
                - Self Lms: si la institución cuenta con un Learning Management System.
                - Device: dispositivo utilizado en clase.
                """
    st.write(texto)

elif seleccion == "Datos":

    with st.expander('Datos sin procesar'):
        st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Datos originales</h1>", unsafe_allow_html=True)
        df = pd.read_csv("../data/raw/students_adaptability.csv")
        st.write(df)
        st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Distribución de features</h1>", unsafe_allow_html=True)
        img = Image.open("../images/dist_variables.png")
        st.image(img)
        st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Distribución de target</h1>", unsafe_allow_html=True)
        img = Image.open("../images/dist_target.png")
        st.image(img)  
  
    with st.expander('Datos procesados'):
        st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Datos procesados</h1>", unsafe_allow_html=True)
        df = pd.read_csv("../data/processed/processed.csv")
        st.write(df)

    with st.expander('Datos para test'):
        st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Datos para test</h1>", unsafe_allow_html=True)
        df = pd.read_csv("../data/test/test.csv")
        st.write(df)

    with st.expander('Datos para train'):
        st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Datos para train</h1>", unsafe_allow_html=True)
        df = pd.read_csv("../data/train/train.csv")
        st.write(df)

elif seleccion == "Oversampling":
    st.write("## Oversampling")
    texto = """
            Cuándo tenemos un sesgo severo en la distribución de clases de nuestros datos de entrenamiento, nos enfrentamos a un problema de clasificación desequilibrada.
            Para contrarestar esta dificultad, tenemos dos técnicas:
            - Oversampling o sobremuestreo : duplicación de muestras de la clase minoritaria
            - Undersampling o submuestreo : eliminar muestras de la clase mayoritaria.

            [Fuente](https://towardsdatascience.com/oversampling-and-undersampling-5e2bbaf56dcf)

            Distribución del target antes y después del oversampling:

    """

    st.write(texto)

    img = Image.open("../images/oversampling.png")
    st.image(img)

elif seleccion == "Modelos de clasificación":
    st.write("## Modelos de clasificación")

    st.write("#### RandomForestClassifier:")
    texto = """
            Un bosque aleatorio es un metaestimador que ajusta una serie de clasificadores de árboles de decisión en varias submuestras del conjunto de datos y utiliza promedios
            para mejorar la precisión predictiva y controlar el sobreajuste.

            [Fuente](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    
            Resultados:
    """
    st.write(texto)
    img = Image.open("../images/matriz_rfc.png")
    st.image(img)

    texto = "Accuracy: 87,4 %"
    st.write(texto)



    st.write("#### GradientBoostingClassifier:")
    texto = """
            Este algoritmo construye un modelo aditivo en etapas avanzadas; permite la optimización de funciones de pérdida diferenciables arbitrarias.

            [Fuente](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
    
            Resultados:
    """
    st.write(texto)
    img = Image.open("../images/matriz_gbc.png")
    st.image(img)

    texto = "Accuracy: 88,1 %"
    st.write(texto)



    st.write("#### SupportVectorClasiffier:")
    texto = """
            El objetivo del algoritmo SVM es encontrar un hiperplano que separe de la mejor forma posible dos clases diferentes de puntos de datos.

            [Fuente](https://es.mathworks.com/discovery/support-vector-machine.html)
    
            Resultados:
    """
    st.write(texto)
    img = Image.open("../images/matriz_svc.png")
    st.image(img)

    texto = "Accuracy: 89,7 %"
    st.write(texto)



    st.write("#### BagginClassifier:")
    texto = """
            Un clasificador Bagging es un metaestimador conjunto que ajusta clasificadores base, cada uno de ellos en subconjuntos aleatorios del conjunto de datos original, y luego agrega sus predicciones individuales (ya sea mediante votación o promediando) para formar una predicción final.

            [Fuente](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
    
            Resultados:
    """
    st.write(texto)
    img = Image.open("../images/matriz_bc.png")
    st.image(img)

    texto = "Accuracy: 86,7 %"
    st.write(texto)


    
    st.write("#### AdaBoostClassifier:")
    texto = """
            Un clasificador AdaBoost es un metaestimador que comienza ajustando un clasificador en el conjunto de datos original y luego ajusta copias adicionales del clasificador en el mismo conjunto de datos, pero donde los pesos de las instancias clasificadas incorrectamente se ajustan de modo que los clasificadores posteriores se centren más en casos difíciles.

            [Fuente](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
    
            Resultados:
    """
    st.write(texto)
    img = Image.open("../images/matriz_abc.png")
    st.image(img)

    texto = "Accuracy: 87,4 %"
    st.write(texto)



    st.write("#### PCA + SVC:")
    texto = """
            Un clasificador AdaBoost es un metaestimador que comienza ajustando un clasificador en el conjunto de datos original y luego ajusta copias adicionales del clasificador en el mismo conjunto de datos, pero donde los pesos de las instancias clasificadas incorrectamente se ajustan de modo que los clasificadores posteriores se centren más en casos difíciles.

            [Fuente](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
    
            Resultados:
    """
    st.write(texto)
    img = Image.open("../images/matriz_gs.png")
    st.image(img)

    texto = "Accuracy: 87,7 %"
    st.write(texto)    

elif seleccion == "Predicción":
    st.write("## Página de Entrenamiento")

    with open('../models/final_model_svc', 'rb') as archivo_entrada:
        modelo_svc = pickle.load(archivo_entrada)

    respuesta_pregunta2 = st.selectbox("Edad del estudiante", [0, 1, 2, 3, 4, 5])
    st.text('"1-5": 0 - "6-10": 1 - "11-15": 2 - "16-20": 3 - "21-25": 4 - "26-30": 5')
    respuesta_pregunta3 = st.selectbox("Nivel educativo", [0, 1, 2])
    st.text('"Escuela": 0 - "Colegio": 1 - "Universidad": 2')
   
    respuesta_pregunta5 = st.selectbox("¿Es el/la estudiante un/a estudiante IT?", [0, 1])
    st.text('"No": 0 - "Si": 1')
    respuesta_pregunta6 = st.selectbox("¿El hogar está dentro del área metropolitana de la ciudad?", [0, 1])
    st.text('"No": 0 - "Si": 1')
    respuesta_pregunta7 = st.selectbox("Nivel de deslastre de carga", [0, 1, 2])
    st.text('"Bajo": 0 - "Moderado": 1 - "Alto": 2')
    respuesta_pregunta8 = st.selectbox("Condición económica del estudiante", [0, 1, 2])
    st.text('"Baja": 0 - "Media": 1 - "Alta": 2')
    respuesta_pregunta9 = st.selectbox("Tipo de conexión a internet", [0, 1, 2])
    st.text('"Conexión por móvil": 0 - "WiFi": 1 - "Otro": 2')
    respuesta_pregunta10 = st.selectbox("Tipo de red", [0, 1, 2])
    st.text('"2G": 0 - "3G": 1 - "4G": 2')
    respuesta_pregunta11 = st.selectbox("Duración (en horas) de las clases", [0, 1, 2])
    st.text('"Menos de 1 hora": 0 - "1-3 horas": 1 - "3-6 horas": 2')
    respuesta_pregunta12 = st.selectbox("¿La institución cuenta con un Learning Management System?", [0, 1])
    st.text('"No": 0 - "Si": 1')
    respuesta_pregunta13 = st.selectbox("Dispositivo utilizado en clase", [0, 1, 2])
    st.text('"Móvil": 0 - "Tablet": 1 - "Computer": 2')
    respuesta_pregunta14 = st.selectbox("Institución gubernamental", [0, 1])
    st.text('"No": 0 - "Si": 1')
    respuesta_pregunta15 = st.selectbox("Institución no gubernamental", [0, 1])
    st.text('"No": 0 - "Si": 1')
    respuesta_pregunta16 = st.selectbox("Género del estudiante Hombre", [0, 1])
    st.text("No: 0 - Si: 1")
    respuesta_pregunta17 = st.selectbox("Género del estudiante Mujer", [0, 1])
    st.text("No: 0 - Si: 1")

    data = {'Age': respuesta_pregunta2, 'Education Level': respuesta_pregunta3,
            'IT Student': respuesta_pregunta5,'Location': respuesta_pregunta6,
              'Load-shedding': respuesta_pregunta7, 'Financial Condition': respuesta_pregunta8,
              'Internet Type': respuesta_pregunta9, 'Network Type': respuesta_pregunta10,
              'Class Duration': respuesta_pregunta11,'Self Lms': respuesta_pregunta12,
              'Device': respuesta_pregunta13, 'Institution_Type_Government': respuesta_pregunta14,
              'Institution_Type_Non Government': respuesta_pregunta15, 'Gender_Boy': respuesta_pregunta16,
              'Gender_Girl': respuesta_pregunta17}
    
    features = pd.DataFrame(data, index=[0])
   
    if st.button('Realizar predicción'):
        st.success(modelo_svc.predict(features)[0])
