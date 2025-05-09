# Documentación del Proyecto: Preparación de Datos para Detección de Daños con Múltiples Etiquetas
---
## TensorFlow/Keras model.

TensorFlow es una biblioteca de código abierto para aprendizaje automático, y Keras es una API de alto nivel que facilita la construcción y entrenamiento de redes neuronales. Cuando trabajamos con imágenes, el flujo general es:

1. Entrada: Se recibe una imagen, por ejemplo, una foto de un auto.
2. Extracción de características: La red neuronal procesa la imagen para extraer características relevantes, como bordes, texturas, formas, etc.
3. Clasificación o predicción: Con base en esas características, la red predice etiquetas o valores, como identificar partes del auto, daños o sugerencias.

## EfficientNetB0

EfficientNetB0 es un modelo de red neuronal convolucional preentrenado en millones de imágenes (ImageNet). Es conocido por ser eficiente en términos de precisión y tamaño, logrando buenos resultados con menos parámetros que otros modelos grandes.

1. Ejemplo visual: Imagina que EfficientNetB0 es un experto que ya sabe reconocer miles de objetos y patrones visuales. Cuando le das una nueva imagen, él extrae las características más importantes sin necesidad de aprender desde cero.
2. En el proyecto: Se usa EfficientNetB0 para que actúe como extractor de características de las imágenes de autos dañados. Esto acelera el entrenamiento y mejora la precisión, porque no se tiene que entrenar un modelo desde cero con muchos datos.

  ### model.summary()

    Model: "functional"

    ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
    ┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
    ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
    │ input_layer_1       │ (None, 224, 224,  │          0 │ -                 │
    │ (InputLayer)        │ 3)                │            │                   │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ efficientnetb0      │ (None, 7, 7,      │  4,049,571 │ input_layer_1[0]… │
    │ (Functional)        │ 1280)             │            │                   │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ global_average_poo… │ (None, 1280)      │          0 │ efficientnetb0[0… │
    │ (GlobalAveragePool… │                   │            │                   │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ dense (Dense)       │ (None, 512)       │    655,872 │ global_average_p… │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ dropout (Dropout)   │ (None, 512)       │          0 │ dense[0][0]       │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ partes (Dense)      │ (None, 22)        │     11,286 │ dropout[0][0]     │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ dannos (Dense)      │ (None, 6)         │      3,078 │ dropout[0][0]     │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ sugerencias (Dense) │ (None, 2)         │      1,026 │ dropout[0][0]     │
    └─────────────────────┴───────────────────┴────────────┴───────────────────┘

    - Total params: 4,720,833 (18.01 MB)
    - Trainable params: 671,262 (2.56 MB)
    - Non-trainable params: 4,049,571 (15.45 MB)

El modelo es una red neuronal construida con TensorFlow/Keras que utiliza una arquitectura llamada EfficientNetB0 como base para extraer características de las imágenes. Esta base es un modelo preentrenado que ya sabe reconocer patrones visuales generales, como bordes, texturas y formas, gracias a haber sido entrenado previamente con una gran cantidad de imágenes.

El modelo muestra la arquitectura y los detalles de los parámetros del modelo TensorFlow/Keras.

1. Nombre del modelo: "functional"
    - Indica que el modelo fue creado usando la API Funcional de Keras.

2. Capas:
    - Cada fila corresponde a una capa en el modelo.
    - Las columnas muestran:
        - Layer (type): El nombre y tipo de la capa.
        - Output Shape: La forma del tensor de salida de esa capa. "None" significa que el tamaño del batch es flexible.
        - Param #: La cantidad de parámetros entrenables (pesos y sesgos) en esa capa.
        - Connected to: Qué capa(s) anterior(es) alimentan esta capa.

3. Detalles de las capas:
    - input_layer_1 (InputLayer): La entrada al modelo, se reciben las imágenes con tamaño 224x224 píxeles y 3 canales de color (RGB).
    - efficientnetb0 (Functional): El modelo base EfficientNetB0, que produce mapas de características con forma (7, 7, 1280). Tiene alrededor de 4 millones de parámetros, esta es la parte principal del modelo que extrae características importantes de la imagen. Por ejemplo, puede detectar que hay una rueda, una puerta o un daño en un auto. Esta capa produce una salida con forma (7, 7, 1280), que es una representación compacta y rica en información de la imagen original. En el modelo, esta parte está congelada, es decir, sus parámetros no se actualizan durante el entrenamiento para aprovechar el conocimiento previo.
    - global_average_pooling2d (GlobalAveragePooling2D): Reduce las dimensiones espaciales (7x7) promediando, Esta capa toma la salida de EfficientNetB0 y la convierte en un vector de 1280 valores, promediando la información espacial. Esto simplifica la información para que las siguientes capas puedan procesarla más fácilmente.
    - dense (Dense): Capa totalmente conectada con 512 neuronas, 655,872 parámetros entrenables que aprende a interpretar las características extraídas para las tareas específicas del modelo. Esta capa sí se entrena y ajusta sus parámetros.
    - dropout (Dropout): Capa de regularización que descarta unidades aleatoriamente durante el entrenamiento para evitar sobreajuste; no tiene parámetros, es decir, usa una técnica de regularización que ayuda a evitar que el modelo se sobreentrene, apagando aleatoriamente algunas neuronas durante el entrenamiento.
    - Tres capas de salida (partes, dannos, sugerencias): Cada una es una capa densa que produce predicciones para diferentes tareas:
        - partes (Dense): Predice 22 posibles etiquetas relacionadas con las partes del objeto (por ejemplo, Capó, Faros izquierdo, Parachoques trasero, etc), con 11,286 parámetros.
        - dannos (Dense): Predice 6 posibles tipos de daños (Abolladura, Deformación, Desprendimiento, Fractura, Rayón y Rotura), con 3,078 parámetros.
        - sugerencias (Dense): Predice 2 posibles sugerencias o recomendaciones (Reparar y/o Reemplazar), con 1,026 parámetros.

4. Resumen de parámetros:
    - Total params: 4,720,833 — número total de parámetros en el modelo.
    - Trainable params: 671,262 — parámetros que se actualizarán durante el entrenamiento (las capas densas y dropout).
    - Non-trainable params: 4,049,571 — parámetros fijos (para los pesos del modelo base EfficientNetB0).

### Resumen:
El modelo utiliza un EfficientNetB0 preentrenado como extractor de características fijo, seguido de capas densas personalizadas para tres tareas de salida diferentes ("partes", "daños" y "sugerencias"). Los parámetros entrenables están principalmente en las capas densas después del modelo base.

El modelo tiene un total de aproximadamente 4.7 millones de parámetros.
De esos, alrededor de 671 mil son entrenables (los que se ajustan durante el entrenamiento).
Los restantes 4 millones son parámetros fijos del EfficientNetB0 preentrenado.
Esta configuración es común en aprendizaje por transferencia, donde se aprovecha un modelo ya entrenado para tareas generales y se adapta a tareas específicas con capas adicionales que sí se entrenan.

---

## 1. Introducción y Marco Teórico

Este proyecto tiene como objetivo preparar y procesar datos para un sistema de detección de daños en partes de vehículos utilizando múltiples etiquetas. Se trabaja con imágenes y etiquetas que describen las piezas del vehículo, tipos de daño y sugerencias de reparación o reemplazo.

Imagina que se tiene una foto de un auto dañado y se quiere que el modelo nos diga:

  - Qué partes del auto están en la imagen (por ejemplo, puerta, capó, parachoques).
  - Qué tipo de daño tiene cada parte (por ejemplo, abolladura, rayón).
  - Qué sugerencias dar para la reparación (por ejemplo, reemplazo, pintura).

El modelo primero usa EfficientNetB0 para entender la imagen en general. Luego, las capas densas interpretan esa información para darte las respuestas específicas en las tres categorías mencionadas.

El enfoque multi-etiqueta permite clasificar simultáneamente múltiples daños y partes afectadas en una imagen, lo que es fundamental para aplicaciones en seguros y evaluación de siniestros automotrices.

¿Por qué aplicar esta arquitectura a tu proyecto?

1. Transferencia de aprendizaje: Se aprovecha un modelo preentrenado para tareas generales y se adapta a mi problema específico, lo que es ideal cuando se tiene un conjunto de datos limitado.
2. Multitarea: El modelo predice simultáneamente varias etiquetas (partes, daños, sugerencias), lo que es eficiente y puede mejorar el rendimiento al compartir representaciones internas.
3. Eficiencia: EfficientNetB0 es ligero y rápido, lo que facilita su uso en aplicaciones prácticas.

---

## 2. Descripción de los Datos

### 2.1 Piezas del Vehículo

Listado completo de las piezas consideradas en el proyecto:

- Antiniebla delantero derecho
- Antiniebla delantero izquierdo
- Capó
- Cerradura capo
- Cerradura maletero
- Cerradura puerta
- Espejo lateral derecho
- Espejo lateral izquierdo
- Faros derecho
- Faros izquierdo
- Guardabarros delantero derecho
- Guardabarros delantero izquierdo
- Guardabarros trasero derecho
- Guardabarros trasero izquierdo
- Luz indicadora delantera derecha
- Luz indicadora delantera izquierda
- Luz indicadora trasera derecha
- Luz indicadora trasera izquierda
- Luz trasera derecho
- Luz trasera izquierdo
- Maletero
- Manija derecha
- Manija izquierda
- Marco de la ventana
- Marco de las puertas
- Moldura capó
- Moldura puerta delantera derecha
- Moldura puerta delantera izquierda
- Moldura puerta trasera derecha
- Moldura puerta trasera izquierda
- Parabrisas delantero
- Parabrisas trasero
- Parachoques delantero
- Parachoques trasero
- Puerta delantera derecha
- Puerta delantera izquierda
- Puerta trasera derecha
- Puerta trasera izquierda
- Rejilla, parrilla
- Rueda
- Tapa de combustible
- Tapa de rueda
- Techo
- Techo corredizo
- Ventana delantera derecha
- Ventana delantera izquierda
- Ventana trasera derecha
- Ventana trasera izquierda
- Ventanilla delantera derecha
- Ventanilla delantera izquierda
- Ventanilla trasera derecha
- Ventanilla trasera izquierda

### 2.2 Tipos de Daño

- Abolladura
- Deformación
- Desprendimiento
- Fractura
- Rayón
- Rotura

### 2.3 Sugerencias

- Reparar
- Reemplazar

---

## 3. Librerías y Configuración del Entorno

Para la preparación y análisis de datos se utilizan las siguientes librerías de Python:

- pandas
- numpy==1.23.5
- scikit-learn
- imblearn
- iterative-stratification
- matplotlib
- scikit-multilearn
- liac-arff
- tensorflow
- tensorflow.keras
- torch
- torchvision
- seaborn
- imblearn
- json

Instalación recomendada:

```bash
pip install --upgrade pip
pip install pandas scikit-learn imblearn iterative-stratification matplotlib scikit-multilearn liac-arff
```

---

## 4. Preparación y Limpieza de Datos

- Se realiza la lectura del archivo CSV con los datos originales.
- Limpieza y estandarización de texto (minúsculas, eliminación de espacios).
- Manejo de valores compuestos en la columna "Tipos de Daño" (por ejemplo, "Abolladura-dent").
- Mapeo de texto a valores numéricos para piezas, daños y sugerencias con manejo de errores.
- Consolidación de etiquetas por imagen para facilitar el procesamiento multi-etiqueta.

---

## 5. Codificación y Consolidación de Etiquetas

- Se codifican las etiquetas de daños, partes y sugerencias en valores numéricos.
- Se agrupan las etiquetas por imagen para obtener listas consolidadas.
- Se guardan los datos procesados en archivos CSV para su uso posterior.

---

## 6. Análisis de Distribución y Manejo de Clases Raras

- Se analiza la distribución de clases multi-etiqueta para identificar clases con pocas muestras.
- Se establecen umbrales para considerar una clase como rara.
- Se agrupan las clases raras bajo una categoría "Otras" para mejorar la robustez del modelo.
- Se actualizan los diccionarios de etiquetas para incluir estas nuevas categorías.
- Se generan archivos de soporte con ejemplos y estadísticas de clases raras.

---

## 7. División de Datos para Entrenamiento, Validación y Prueba

- Se utiliza la técnica de estratificación multilabel para mantener la distribución de clases en los conjuntos.
- División recomendada:
  - 70% entrenamiento
  - 15% validación
  - 15% prueba
- Se guardan los conjuntos en archivos CSV separados.
- Se almacenan metadatos para reproducibilidad y trazabilidad.

---

## 8. Mejores Prácticas y Sugerencias

- Mantener la reproducibilidad mediante el uso de semillas aleatorias y guardado de metadatos.
- Analizar y manejar clases raras para evitar sesgos en el modelo.
- Utilizar técnicas de estratificación multilabel para preservar la distribución de etiquetas.
- Documentar claramente cada paso del procesamiento para facilitar mantenimiento y mejoras.
- Realizar análisis exploratorios para entender la distribución y calidad de los datos.

---

## 9. Conclusiones

La preparación cuidadosa y estructurada de los datos es fundamental para el éxito de modelos de detección multi-etiqueta. Este proyecto proporciona un pipeline completo desde la limpieza, codificación, análisis, agrupamiento de clases raras y división estratificada, asegurando datos de calidad para el entrenamiento y evaluación.

---

## 10. ¿Qué otras alternativas podrías considerar?

    1. Otros modelos preentrenados:
        - ResNet, VGG, MobileNet, DenseNet, etc.
        - Por ejemplo, MobileNet es aún más ligero y puede ser útil para dispositivos con recursos limitados.

    2. Entrenamiento desde cero:
        - Si tienes muchos datos específicos, podrías entrenar un modelo personalizado sin usar preentrenamiento, pero esto requiere más tiempo y recursos.

    3. Modelos especializados para multitarea:
        - Arquitecturas diseñadas para manejar múltiples salidas con diferentes tipos de datos o tareas.

    4. Ajuste fino (fine-tuning):
        - En lugar de congelar EfficientNetB0, El ajuste fino consiste en "descongelar" algunas de las capas superiores del modelo preentrenado (en este caso EfficientNetB0) para que sus pesos se actualicen durante el entrenamiento. Esto permite que el modelo adapte sus características generales a las particularidades de el problema, mejorando potencialmente el rendimiento.

---

## 11. Bibliografía

- Documentación oficial de pandas, scikit-learn, imblearn y matplotlib.
- Artículos y recursos sobre clasificación multilabel y estratificación multilabel.
- Referencias específicas pueden añadirse según fuentes utilizadas.

---

## 12. Extras

### Mapa Mental (Mind Map)

> [Incluir aquí un diagrama o enlace a un mapa mental que resuma el flujo del proyecto]

### Imágenes y Gráficos

> [Incluir gráficos de distribución de clases, diagramas de flujo y otros recursos visuales relevantes]

---

*Fin de la documentación del proyecto.*
