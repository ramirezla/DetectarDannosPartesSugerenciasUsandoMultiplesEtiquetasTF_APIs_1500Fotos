1. Explicación de los gráficos:
    - El eje X representa el recall (sensibilidad), que mide la proporción de verdaderos positivos detectados sobre el total de positivos reales.
    - El eje Y representa la precisión, que mide la proporción de verdaderos positivos sobre el total de predicciones positivas realizadas.
    - Cada curva muestra cómo varían precisión y recall al cambiar el umbral de clasificación para esa clase.
    - Un área bajo la curva (Average Precision, AP) más alta indica mejor desempeño para esa clase.
    - Curvas que se mantienen altas en precisión y recall indican que el modelo puede distinguir bien esa clase.
    - Curvas bajas o muy planas indican dificultad para detectar esa clase correctamente.

2. Interpretación práctica:
    - Puedes usar estas curvas para elegir umbrales personalizados para cada clase que optimicen el balance entre precisión y recall según tus necesidades.
    - Clases con AP bajo pueden requerir más datos, mejor arquitectura o técnicas de balanceo.
    - Las curvas ayudan a entender qué clases el modelo predice bien y cuáles no.