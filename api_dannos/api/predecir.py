
import numpy as np
from typing import Dict, List, Union
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: todos, 1: info, 2: warnings, 3: errors  Suprimir errores de tensorflow
import tensorflow as tf
from tensorflow.keras import applications

class DamagePredictor:
    def __init__(self, model_path: str):
        """Inicializa el predictor con el modelo cargado"""
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = (224, 224)
        
        # Diccionarios de etiquetas (deben coincidir con el entrenamiento)
        self.label_maps = {
            'partes': {
                1: "Antiniebla delantero derecho",
                # ... (completar con tus diccionarios)
            },
            'dannos': {
                1: "Abolladura",
                # ... (completar)
            },
            'sugerencias': {
                1: "Reparar",
                # ... (completar)
            }
        }

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocesamiento para EfficientNet"""
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = applications.efficientnet.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_path: str) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """Predicción principal"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Archivo no encontrado: {image_path}")
        
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array)
        
        return self._format_predictions(predictions)

    def _format_predictions(self, predictions: List[np.ndarray]) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """Formatea las predicciones para la API"""
        results = {}
        for i, category in enumerate(['partes', 'dannos', 'sugerencias']):
            probs = predictions[i][0]
            top_indices = np.argsort(probs)[::-1][:3]  # Top 3 predicciones
            
            results[category] = [{
                "label": self.label_maps[category].get(idx+1, f"Clase_{idx+1}"),
                "probability": float(probs[idx])
            } for idx in top_indices]
        
        return results