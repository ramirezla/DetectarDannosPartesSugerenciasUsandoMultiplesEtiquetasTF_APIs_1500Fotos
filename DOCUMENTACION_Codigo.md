# Documentación del Código: PreparacionDeDatosMultiplesEtiquetasTF_API.ipynb

---

## Funciones Documentadas

### 1. limpiar_texto(texto)

- **Descripción:** Limpia y estandariza el texto para comparación.
- **Parámetros:**
  - `texto` (str): Texto a limpiar.
- **Retorna:** Texto en minúsculas y sin espacios al inicio o final. Si el texto es NaN (Nulo), retorna cadena vacía.
- **Uso:** Se utiliza para normalizar cadenas antes de compararlas o mapearlas.

---

### 2. procesar_dano(texto)

- **Descripción:** Procesa la columna "Tipos de Daño" para manejar valores compuestos (ejemplo: "Abolladura-dent").
- **Parámetros:**
  - `texto` (str): Texto del tipo de daño.
- **Retorna:** La parte principal del daño antes del guion, o el texto original si no contiene guion.
- **Uso:** Normaliza los valores de daño para facilitar el mapeo.

---

### 3. mapear_valor(texto, diccionario, columna)

- **Descripción:** Mapea un texto a un valor numérico usando un diccionario, con manejo de errores.
- **Parámetros:**
  - `texto` (str): Texto a mapear.
  - `diccionario` (dict): Diccionario de mapeo texto -> número.
  - `columna` (str): Nombre de la columna para aplicar reglas específicas (ej. "Tipos de Daño").
- **Retorna:** Valor numérico correspondiente o -1 si no se encuentra o hay error.
- **Uso:** Codifica las etiquetas de texto en valores numéricos para procesamiento.

---

### 4. consolidar_etiquetas(group)

- **Descripción:** Consolida etiquetas de daños, partes y sugerencias agrupadas por imagen.
- **Parámetros:**
  - `group` (DataFrame): Grupo de filas correspondiente a una imagen.
- **Retorna:** Serie con listas de etiquetas consolidadas para 'dannos', 'partes' y 'sugerencias'.
- **Uso:** Agrupa etiquetas para cada imagen en listas para facilitar el manejo multi-etiqueta.

---

### 5. convert_string_lists(df)

- **Descripción:** Convierte columnas con listas en formato string a listas reales de Python.
- **Parámetros:**
  - `df` (DataFrame): DataFrame con columnas a convertir.
- **Retorna:** DataFrame con columnas convertidas.
- **Uso:** Prepara los datos para procesamiento posterior asegurando el tipo correcto.

---

### 6. analyze_multilabel_class_distribution(y, class_names=None)

- **Descripción:** Analiza y grafica la distribución de clases en una matriz binaria multilabel.
- **Parámetros:**
  - `y` (numpy array): Matriz binaria de etiquetas multilabel (n_samples, n_classes).
  - `class_names` (list, opcional): Nombres de las clases.
- **Retorna:** None (imprime y grafica resultados).
- **Uso:** Identifica clases con pocas muestras para análisis y balanceo.

---

### 7. cargar_y_preparar_datos(ruta_encoded)

- **Descripción:** Carga y prepara datos codificados desde un archivo CSV.
- **Parámetros:**
  - `ruta_encoded` (str): Ruta al archivo CSV codificado.
- **Retorna:** DataFrame con listas convertidas a objetos Python.
- **Uso:** Prepara datos para procesamiento y agrupamiento.

---

### 8. identificar_clases_raras(series_etiquetas, umbral)

- **Descripción:** Identifica clases con menos muestras que un umbral dado.
- **Parámetros:**
  - `series_etiquetas` (Series): Serie con listas de etiquetas.
  - `umbral` (int): Número mínimo de muestras para no ser considerada rara.
- **Retorna:** Conjunto de clases raras.
- **Uso:** Detecta clases poco representadas para agrupamiento.

---

### 9. agrupar_clases_raras(etiquetas, clases_raras)

- **Descripción:** Reemplaza etiquetas raras por una clase agrupada (ej. 999).
- **Parámetros:**
  - `etiquetas` (list): Lista de etiquetas.
  - `clases_raras` (set): Conjunto de clases raras.
- **Retorna:** Lista con etiquetas raras reemplazadas.
- **Uso:** Simplifica el conjunto de clases para mejorar el entrenamiento.

---

### 10. procesar_y_agrupar(ruta_entrada, ruta_salida)

- **Descripción:** Proceso completo para cargar datos, identificar y agrupar clases raras, y guardar resultados.
- **Parámetros:**
  - `ruta_entrada` (str): Ruta al archivo CSV codificado original.
  - `ruta_salida` (str): Ruta para guardar el archivo agrupado.
- **Retorna:** DataFrame con datos agrupados.
- **Uso:** Pipeline para manejo de clases raras y preparación de datos.

---

### 11. analizar_distribucion(df, tipo='partes')

- **Descripción:** Analiza y muestra la distribución de clases para un tipo dado ('partes' o 'dannos').
- **Parámetros:**
  - `df` (DataFrame): DataFrame con datos.
  - `tipo` (str): Tipo de etiqueta a analizar ('partes' o 'dannos').
- **Retorna:** None (imprime y grafica resultados).
- **Uso:** Visualiza la distribución para detectar desequilibrios.

---

### 12. verificar_agrupamiento(df_original, df_agrupado)

- **Descripción:** Compara los DataFrames antes y después del agrupamiento de clases raras.
- **Parámetros:**
  - `df_original` (DataFrame): Datos originales.
  - `df_agrupado` (DataFrame): Datos con clases agrupadas.
- **Retorna:** None (imprime resultados de verificación).
- **Uso:** Valida que el agrupamiento se haya realizado correctamente.

---

### 13. check_distribution(df, name)

- **Descripción:** Verifica la distribución de etiquetas en un conjunto de datos.
- **Parámetros:**
  - `df` (DataFrame): DataFrame con datos.
  - `name` (str): Nombre del conjunto (ej. "Entrenamiento").
- **Retorna:** None (imprime resumen de distribución).
- **Uso:** Control de calidad para conjuntos de entrenamiento, validación y prueba.

---

### 14. convert_to_serializable(obj)

- **Descripción:** Convierte objetos numpy y otros tipos a tipos nativos de Python para serialización JSON.
- **Parámetros:**
  - `obj` (varios): Objeto a convertir.
- **Retorna:** Objeto convertido.
- **Uso:** Facilita el guardado de metadatos en formato JSON.

---

### 15. get_rare_classes(df, column, threshold=5)

- **Descripción:** Obtiene clases raras en una columna específica según un umbral.
- **Parámetros:**
  - `df` (DataFrame): DataFrame con datos.
  - `column` (str): Columna a analizar.
  - `threshold` (int): Umbral para considerar clase rara.
- **Retorna:** Lista de clases raras.
- **Uso:** Identificación para manejo especial de clases.

---

### 16. create_support_files(df, split_name)

- **Descripción:** Crea archivos JSON con soporte para clases raras y ejemplos.
- **Parámetros:**
  - `df` (DataFrame): DataFrame con datos.
  - `split_name` (str): Nombre del conjunto (ej. "train").
- **Retorna:** None (genera archivos en disco).
- **Uso:** Documentación y soporte para análisis y balanceo.

---

*Fin de la documentación del código.*
