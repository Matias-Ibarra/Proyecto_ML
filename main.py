import streamlit as st
import pandas as pd
from PIL import Image
import streamlit.components.v1 as c

st.set_page_config(page_title="Cargadores",
                   page_icon=":electric_plug:")

color_de_fondo = "#363636"

seleccion = st.sidebar.selectbox("Selecciona menu", ['Home','Datos', 'Modelos de machine learning'])


if seleccion == "Home":
    st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Adaptabilidad a la educación en línea</h1>", unsafe_allow_html=True)
    img = Image.open("./images/alumno.jpg")
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
   
#     with st.expander("Introducción del proyecto"):
#         pass

        

    # with st.expander("Datos"):
    #     df = pd.read_csv("./data/raw/students_adaptability.csv")
    #     st.write(df.head())

    # with st.expander("Datos"):
    #     df = pd.read_csv("./data/raw/students_adaptability.csv")
    #     st.write(df.head())

elif seleccion == "Datos":

    # opciones = ['Datos sin procesar', 'Datos procesados', 'Datos para test', 'Datos para train']
    # seleccion_2 = st.multiselect("Selecciona opciones:", opciones)
    # st.write("Opciones seleccionadas:", seleccion)

    with st.expander('Datos sin procesar'):
        st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Datos originales</h1>", unsafe_allow_html=True)
        df = pd.read_csv("./data/raw/students_adaptability.csv")
        st.write(df)
        st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Distribución de features</h1>", unsafe_allow_html=True)
        img = Image.open("./images/dist_variables.png")
        st.image(img)
        st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Distribución de target</h1>", unsafe_allow_html=True)
        img = Image.open("./images/dist_target.png")
        st.image(img)  

    # filtro = st.sidebar.selectbox("Selecciona un distrito", df['DISTRITO'].unique())
    # df_filtered = df[df['DISTRITO']==filtro]
  
    with st.expander('Datos procesados'):
        st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Datos procesados</h1>", unsafe_allow_html=True)
        df = pd.read_csv("./data/processed/processed.csv")
        st.write(df)

    with st.expander('Datos para test'):
        st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Datos para test</h1>", unsafe_allow_html=True)
        df = pd.read_csv("./data/test/test.csv")
        st.write(df)

    with st.expander('Datos para train'):
        st.markdown("<h1 style='text-align: center; background-color: #363636; color: white;'>Datos para train</h1>", unsafe_allow_html=True)
        df = pd.read_csv("./data/train/train.csv")
        st.write(df)
      

elif seleccion == "Datos procesados":
    pass
    
    # file = open("data/heatmap.html", "r")
    # c.html(file.read(), height=400)

    # df_filtered.rename(columns={"latidtud":"lat", "longitud":"lon"}, inplace=True)
    # st.write(df)

    # st.map(df_filtered)

    # filtro_2 = st.sidebar.radio("Elige el nº de cargadores", [1,2,3,4])

#     st.sidebar.button("Click aquí")