# **Adaptabilidad a la educación online**
***
#### *Matias Ibarra - Bootcamp Data Science - The Bridge School - Año 2023*
***
<img src="./images/alumno.jpg" width ="800" height="400">

El siguiente proyecto tiene como objetivo predecir cómo se adaptarán los alumnos a las clases online en base a una serie de atributos. 

***
## Fuente de datos:
**Los datos para el proyecto fueron extraidos de la web de [Kaggle](https://www.kaggle.com/datasets/mdmahmudulhasansuzan/students-adaptability-level-in-online-education)**

La variable target es la columna 'Adaptivity Level' que indica el nivel en el que los alumnos se adaptan a las clases online:
Los valores de target son:
* Low: adaptabilidad baja
* Moderate: adaptabilidad moderada
* High: adaptabilidad alta

### Análisis de variables:
* Gender: género del estudiante.
* Age: rango de edad.
* Education Level: nivel de la Institución Educativa.
* Institution Type: si la Institución es o no gubernamental.
* IT Student: si es o no un alumno IT.
* Location: si el alumno vive en la ciudad o no.
* Load-shedding: nivel de deslastre de carga.
* Financial Condition: condición financiera de la familia del estudiante.
* Internet Type: tipo de conexión a internet.
* Network Type: tipo de red.
* Class Duration: duración diaria de las clases.
* Self Lms: si la institución cuenta con un Learning Management System.
* Device: dispositivo utilizado en clase.

### Índice:
* [app](./app/): archivo para ejecutar streamlit y los requerimientos.
* [data](./data/): datos procesados y sin procesar, tambien están separados los datos finales de entrenamiento y test.
* [docs](./docs/): memoria del proyecto donde se resumen todos los pasos y presentación en powerpoint.
* [images](./images/): imágenes utilizadas.
* [models](./models/): modelos de machine learning utilizados.
* [notebooks](./notebooks/): notebooks de fuente de datos, EDA y entrenamiento-evaluación.
* [src](./src/): archivos py de procesamiento de datos, evaluación y entrenamiento.
