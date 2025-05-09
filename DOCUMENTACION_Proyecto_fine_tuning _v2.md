# Documentación del Proyecto: Preparación de Datos para Detección de Daños con Múltiples Etiquetas TensorFlow/keras usando fine-tuning
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

## Ajuste fino (fine-tuning):
    - En lugar de congelar EfficientNetB0, El ajuste fino consiste en "descongelar" algunas de las capas superiores del modelo preentrenado (en este caso EfficientNetB0) para que sus pesos se actualicen durante el entrenamiento. Esto permite que el modelo adapte sus características generales a las particularidades de el problema, mejorando potencialmente el rendimiento.

### Implementar ajuste fino en el proyecto

    1. Carga el modelo base EfficientNetB0 con pesos preentrenados y sin la capa superior (include_top=False):

        from tensorflow.keras.applications import EfficientNetB0

        base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    2. Congela todas las capas inicialmente para entrenar solo las capas superiores personalizadas:

        base_model.trainable = False

    3. Agrega tus capas personalizadas (global average pooling, dense, dropout, salidas):

    from tensorflow.keras import layers, Model, Input

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    partes_output = layers.Dense(22, activation='sigmoid', name='partes')(x)
    dannos_output = layers.Dense(6, activation='sigmoid', name='dannos')(x)
    sugerencias_output = layers.Dense(2, activation='sigmoid', name='sugerencias')(x)

    model = Model(inputs, [partes_output, dannos_output, sugerencias_output])

    4. Entrena el modelo con el modelo base congelado para que las capas superiores aprendan primero:

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=initial_epochs, validation_data=val_data)

    5. Descongela algunas capas superiores del modelo base para ajuste fino: Por ejemplo, descongelar las últimas 20 capas:

    base_model.trainable = True

    # Congelar todas las capas excepto las últimas 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    6. Compila el modelo con una tasa de aprendizaje más baja para evitar grandes cambios bruscos:

    from tensorflow.keras.optimizers import Adam

    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    7. Continúa entrenando el modelo para ajustar las capas descongeladas:

    model.fit(train_data, epochs=fine_tune_epochs, validation_data=val_data)

### Mejoras emplementadas en el código para el ajuste fino

1. Ajuste de las tasas de aprendizaje:
    - Se redujo la tasa de aprendizaje inicial de 1e-4 a 5e-5 para un entrenamiento más estable y menos agresivo.
    - Se redujo la tasa de aprendizaje para el fine-tuning de 1e-5 a 5e-6 para evitar grandes cambios en los pesos durante el ajuste fino, lo que ayuda a preservar el conocimiento previo del modelo base.

2. Aumento del número de capas descongeladas en el fine-tuning:
    - Se pasó de descongelar las últimas 20 capas a las últimas 30 capas del modelo base EfficientNetB0.
    - Esto permite que el modelo ajuste más parámetros durante el fine-tuning, mejorando su capacidad de adaptación a los datos específicos, pero manteniendo congeladas las capas más generales para evitar sobreajuste.

3. Reducción del dropout de 0.5 a 0.3:
    - Se disminuyó la tasa de dropout para reducir la regularización.
    - Esto puede ayudar a que el modelo aprenda patrones más complejos si no hay un problema severo de sobreajuste, mejorando la capacidad de generalización.

Estas mejoras buscan un balance entre la estabilidad del entrenamiento y la capacidad del modelo para adaptarse mejor a los datos específicos, con la expectativa de mejorar la precisión especialmente en las categorías donde el desempeño era limitado.

### Consideraciones importantes:

        - El número de capas a descongelar depende de tu conjunto de datos y recursos computacionales.
        - Usar una tasa de aprendizaje baja durante el ajuste fino es crucial para evitar que los pesos preentrenados se modifiquen demasiado rápido.
        - Monitorea el rendimiento en el conjunto de validación para evitar sobreajuste.
        - Puedes experimentar descongelando más o menos capas según los resultados.

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
    - Total params: 8,765,279 (33.44 MB)
    - Trainable params: 2,022,222 (7.71 MB)
    - Non-trainable params: 2,698,611 (10.29 MB)
    - Optimizer params: 4,044,446 (15.43 MB)


El modelo es un modelo funcional de Keras con múltiples salidas (multi-output), diseñado para clasificar imágenes en tres categorías diferentes: partes, daños y sugerencias.

1. Capas del modelo:
    - input_layer_1 (InputLayer): Es la capa de entrada que recibe imágenes con tamaño 224x224 píxeles y 3 canales de color (RGB).

    - efficientnetb0 (Functional): Es la base del modelo, una red EfficientNetB0 preentrenada en ImageNet, que extrae características relevantes de las imágenes. Su salida es un tensor de forma (None, 7, 7, 1280), donde None es el tamaño variable del batch.

    - global_average_pooling2d (GlobalAveragePooling2D): Esta capa reduce la salida del EfficientNetB0 promediando cada mapa de características, transformando la salida a un vector de 1280 elementos por imagen.

    - dense (Dense): Una capa densa con 512 neuronas y función de activación ReLU que aprende representaciones más complejas a partir del vector de características.

    - dropout (Dropout): Capa que ayuda a evitar el sobreajuste apagando aleatoriamente el 50% de las neuronas durante el entrenamiento.

    - partes (Dense): Capa de salida para predecir las partes del vehículo, con 22 neuronas (una por cada clase de parte) y función de activación sigmoide para clasificación multilabel.

    - dannos (Dense): Capa de salida para predecir los tipos de daños, con 6 neuronas y activación sigmoide.

    - sugerencias (Dense): Capa de salida para predecir sugerencias (reparar o reemplazar), con 2 neuronas y activación sigmoide.

2. Parámetros:
    - El modelo tiene un total de 8,765,279 parámetros.

    - De estos, 2,022,222 son parámetros entrenables (los que se ajustan durante el entrenamiento).

    - 2,698,611 son parámetros no entrenables (pesos fijos del modelo base EfficientNetB0 congelado inicialmente).

    - 4,044,446 son parámetros del optimizador (relacionados con el algoritmo de optimización).

3. Funcionamiento general:
    - El modelo toma una imagen de entrada y la procesa a través de EfficientNetB0 para extraer características.

    - Luego, esas características se resumen y pasan por capas densas para aprender patrones específicos.

    - Finalmente, el modelo produce tres salidas simultáneas que predicen las etiquetas para partes, daños y sugerencias.


    ### Evaluando el modelo:

    6/6 ━━━━━━━━━━━━━━━━━━━━ 3s 456ms/step

    - dannos_accuracy: 0.5638 
    - dannos_loss: 0.3415 
    - loss: 0.8582 
    - partes_accuracy: 0.3510 
    - partes_loss: 0.1961 
    - sugerencias_accuracy: 0.8775 
    - sugerencias_loss: 0.3190

    Pérdida total:          0.8709885478019714
    Pérdida partes:         0.2027958631515503 - Accuracy partes: 0.5722543597221375
    Pérdida daños:          0.35367903113365173 - Accuracy daños: 0.35260117053985596
    Pérdida sugerencias:    0.30909252166748047 - Accuracy sugerencias: 0.8786126971244812

    Generando predicciones...
    6/6 ━━━━━━━━━━━━━━━━━━━━ 5s 629ms/step

        Reporte para Partes:
                                            precision    recall  f1-score   support

                                Capó            0.00      0.00      0.00        12
                        Faros derecho           0.00      0.00      0.00        11
                        Faros izquierdo         0.00      0.00      0.00         9
        Guardabarros delantero derecho          0.00      0.00      0.00        16
        Guardabarros delantero izquierdo        0.00      0.00      0.00        17
            Guardabarros trasero derecho        0.00      0.00      0.00         8
        Guardabarros trasero izquierdo          0.00      0.00      0.00        11
                    Luz trasera derecho         0.00      0.00      0.00         4
                Luz trasera izquierdo           0.00      0.00      0.00         7
                                Maletero        0.00      0.00      0.00         5
                    Marco de las puertas        0.00      0.00      0.00         8
                    Parabrisas delantero        0.90      0.75      0.82        24
                    Parabrisas trasero          0.00      0.00      0.00         4
        ...
        macro avg           0.88      0.87      0.87       204
        weighted avg        0.89      0.90      0.89       204
        samples avg         0.91      0.92      0.90       204

- La precisión (accuracy) para la categoría "sugerencias" es alta (~87-88%), lo que indica que el modelo predice bien esta salida.
- La precisión para "partes" es moderada (~35-57%), mostrando que el modelo tiene dificultades para identificar correctamente las partes del vehículo.
- La precisión para "daños" es baja (~35-56%), indicando que el modelo tiene un desempeño limitado en la detección de daños.
- Las pérdidas (loss) para cada salida reflejan la dificultad del modelo para ajustarse a los datos, siendo la pérdida total alrededor de 0.87.
- El reporte de clasificación muestra que para muchas clases de partes la precisión, recall y f1-score son cero, lo que sugiere que el modelo no está detectando esas clases específicas.
- Sin embargo, algunas clases como "Parabrisas delantero" tienen buenos valores de precisión y recall, lo que indica que el modelo puede identificar correctamente ciertas clases.

### Resumen:
En resumen, el modelo funciona bien para la predicción de sugerencias, pero tiene margen de mejora para las predicciones de partes y daños. Esto puede deberse a la complejidad del problema, la cantidad y calidad de datos, o la necesidad de ajustar hiperparámetros o arquitectura.

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

1. Implementar métricas adicionales:
    - Precisión, recall y F1-score para cada salida (partes, daños, sugerencias) durante el entrenamiento.
    - Métricas multilabel específicas como AUC-ROC o Average Precision.
    - Usar callbacks personalizados para monitorear estas métricas.

2. Mejorar el modelo:
    - Ajustar hiperparámetros como tasa de aprendizaje, número de capas descongeladas en el fine-tuning, tamaño del dropout.
    - Probar arquitecturas base diferentes (por ejemplo, EfficientNetB3 o ResNet50).
    - Incrementar el tamaño y calidad del dataset con más imágenes etiquetadas.
    - Aplicar técnicas de aumento de datos más variadas.

3. Balancear clases:
    - Usar pesos de clase para manejar desequilibrios en las etiquetas.
    - Filtrar o agrupar clases con muy pocos ejemplos.
    - Evaluar con validación cruzada para obtener métricas más robustas.

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
