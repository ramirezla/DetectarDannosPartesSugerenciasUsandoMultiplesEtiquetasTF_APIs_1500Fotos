## API para Predicción de Daños en Vehículos

1. Estructura del proyecto

        api_dannos/                       # Directorio raíz del proyecto
            ├── api/                          # Paquete principal
            │   ├── __init__.py               # Archivo de inicialización
            │   ├── app.py                    # Aplicación FastAPI
            │   ├── predecir.py               # Lógica de predicción
            │   └── requirements.txt          # Dependencias
            ├── modelos/
            │   └── final_model.keras         # Modelo entrenado
            └── predecir/                     # Carpeta para imágenes temporales

        api_dannos/                           # Directorio raíz del proyecto
            ├── main.py                       # Principal
            ├── api/                          # Paquete principal
            │   ├── __init__.py               # Archivo de inicialización
            │   └── app.py                    # Aplicación FastAPI
            ├── modelos/
            │   └── final_model.keras         # Modelo entrenado
            ├── predecir/                     # Carpeta para imágenes temporales
            └── requirements.txt              # Dependencias

# Desde la raíz del proyecto (api_dannos/)
uvicorn api.app:app --reload --host 0.0.0.0 --port 5000


```python
curl -X POST -F "file=@/data/Python/data/predecir/rayon_02.jpg" http://localhost:5000/predict
```

2. Implementación del API

model_loader.py


```python
import tensorflow as tf
from tensorflow.keras import applications
import numpy as np
import os

class DamagePredictor:
    def __init__(self, model_path):
        """Carga el modelo y los diccionarios de etiquetas"""
        self.model = tf.keras.models.load_model(model_path)
        self.label_to_cls_piezas = {
            1: "Antiniebla delantero derecho",
            # ... (todo tu diccionario de piezas)
        }
        self.label_to_cls_danos = {
            1: "Abolladura",
            # ... (todo tu diccionario de daños)
        }
        self.label_to_cls_sugerencias = {
            1: "Reparar",
            # ... (todo tu diccionario de sugerencias)
        }

    def preprocess_image(self, image_path, img_size=(224, 224)):
        """Preprocesa la imagen para el modelo"""
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = applications.efficientnet.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_path):
        """Realiza la predicción sobre una imagen"""
        # Preprocesar imagen
        img_array = self.preprocess_image(image_path)
        
        # Hacer predicción
        predictions = self.model.predict(img_array)
        
        # Procesar resultados
        def get_top_predictions(classes, probs, label_dict, top_n=3):
            top_items = sorted(zip(classes, probs[0]), key=lambda x: x[1], reverse=True)[:top_n]
            return [{"label": label_dict.get(int(cls), f"Clase_{int(cls)}"), "probability": float(prob)} for cls, prob in top_items]
        
        return {
            "partes": get_top_predictions(range(len(self.label_to_cls_piezas)), predictions[0], self.label_to_cls_piezas),
            "dannos": get_top_predictions(range(len(self.label_to_cls_danos)), predictions[1], self.label_to_cls_danos),
            "sugerencias": get_top_predictions(range(len(self.label_to_cls_sugerencias)), predictions[2], self.label_to_cls_sugerencias)
        }
```

app.py


```python
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from model_loader import DamagePredictor

# Configuración
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = '../models/final_model.keras'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo al iniciar
predictor = DamagePredictor(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar si se envió un archivo
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Verificar si se seleccionó un archivo
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Realizar predicción
            result = predictor.predict(filepath)
            
            # Eliminar archivo temporal
            os.remove(filepath)
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    # Crear carpeta de uploads si no existe
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
```

requirements.txt


```python
flask==2.3.2
tensorflow==2.10.1
numpy==1.23.5
werkzeug==2.3.7
```

3. Instrucciones para usar el API

    Instalar dependencias:


```python
pip install -r requirements.txt
```

Ejecutar el API:


```python
python app.py
```

Enviar solicitudes:


```python
import requests

url = 'http://localhost:5000/predict'
files = {'file': open('ruta/a/tu/imagen.jpg', 'rb')}
response = requests.post(url, files=files)

print(response.json())
```

Desde cURL:


```python
curl -X POST -F "file=@ruta/a/tu/imagen.jpg" http://localhost:5000/predict
```

4. Respuesta del API

La API devolverá un JSON con el siguiente formato:


```python
{
  "partes": [
    {"label": "Puerta delantera derecha", "probability": 0.95},
    {"label": "Parachoques delantero", "probability": 0.87},
    {"label": "Faros derecho", "probability": 0.62}
  ],
  "dannos": [
    {"label": "Abolladura", "probability": 0.92},
    {"label": "Rayón", "probability": 0.85},
    {"label": "Deformación", "probability": 0.41}
  ],
  "sugerencias": [
    {"label": "Reparar", "probability": 0.78},
    {"label": "Reemplazar", "probability": 0.22}
  ]
}
```

5. Consideraciones adicionales

    - Seguridad:
        - Valida siempre las extensiones de archivo
        - Usa secure_filename para evitar path traversal
        - Considera añadir autenticación para producción

    - Rendimiento:
        - El modelo se carga una sola vez al iniciar el servidor
        - Las imágenes se eliminan después de procesarlas

    - Producción:
        - Usa Gunicorn o Waitress para servir la aplicación
        - Considera usar un servidor Nginx como reverse proxy
        - Implementa logging para monitorear el servicio

Esta implementación proporciona una API robusta y lista para producción que puede recibir imágenes de vehículos dañados y devolver predicciones detalladas sobre las partes afectadas, tipos de daño y sugerencias de reparación.
