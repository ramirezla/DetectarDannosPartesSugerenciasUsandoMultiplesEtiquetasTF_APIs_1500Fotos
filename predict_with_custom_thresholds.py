import os
import json
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import applications

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El archivo del modelo no existe: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Modelo cargado desde {model_path}")
    return model

def load_mlb_objects():
    with open("mlb_partes.pkl", "rb") as f:
        mlb_partes = pickle.load(f)
    with open("mlb_dannos.pkl", "rb") as f:
        mlb_dannos = pickle.load(f)
    with open("mlb_sugerencias.pkl", "rb") as f:
        mlb_sugerencias = pickle.load(f)
    return mlb_partes, mlb_dannos, mlb_sugerencias

def load_thresholds(thresholds_path):
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"El archivo de umbrales no existe: {thresholds_path}")
    with open(thresholds_path, 'r') as f:
        thresholds = json.load(f)
    return thresholds

def preprocess_image(image_path, img_size=(224, 224)):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_with_thresholds(image_path, model, mlb_partes, mlb_dannos, mlb_sugerencias, thresholds_partes, img_size=(224, 224)):
    img_array = preprocess_image(image_path, img_size)
    predictions = model.predict(img_array)

    partes_probs = predictions[0][0]
    dannos_probs = predictions[1][0]
    sugerencias_probs = predictions[2][0]

    # Aplicar umbrales personalizados para partes
    partes_pred = []
    for i, cls in enumerate(mlb_partes.classes_):
        cls_name = str(cls)
        threshold = thresholds_partes.get(cls_name, 0.5)  # usar 0.5 si no está definido
        partes_pred.append((cls_name, partes_probs[i], partes_probs[i] >= threshold))

    # Para daños y sugerencias se usa umbral fijo 0.5 (puede extenderse si se desea)
    dannos_pred = [(str(cls), dannos_probs[i], dannos_probs[i] >= 0.5) for i, cls in enumerate(mlb_dannos.classes_)]
    sugerencias_pred = [(str(cls), sugerencias_probs[i], sugerencias_probs[i] >= 0.5) for i, cls in enumerate(mlb_sugerencias.classes_)]

    return {
        'partes': partes_pred,
        'dannos': dannos_pred,
        'sugerencias': sugerencias_pred
    }

def print_predictions(predictions, label_to_cls_piezas, label_to_cls_danos, label_to_cls_sugerencias):
    print("\nPredicciones para la imagen con umbrales personalizados:\n")
    print("Partes:")
    # Ordenar de mayor a menor probabilidad
    partes_sorted = sorted(predictions['partes'], key=lambda x: x[1], reverse=True)
    for cls_id, prob, pred in partes_sorted:
        name = label_to_cls_piezas.get(int(cls_id), f"Clase_{cls_id}")
        print(f"- {name}: {prob:.2%} - {'Positivo' if pred else 'Negativo'}")

    print("\nDaños:")
    daños_sorted = sorted(predictions['dannos'], key=lambda x: x[1], reverse=True)
    for cls_id, prob, pred in daños_sorted:
        name = label_to_cls_danos.get(int(cls_id), f"Clase_{cls_id}")
        print(f"- {name}: {prob:.2%} - {'Positivo' if pred else 'Negativo'}")

    print("\nSugerencias:")
    sugerencias_sorted = sorted(predictions['sugerencias'], key=lambda x: x[1], reverse=True)
    for cls_id, prob, pred in sugerencias_sorted:
        name = label_to_cls_sugerencias.get(int(cls_id), f"Clase_{cls_id}")
        print(f"- {name}: {prob:.2%} - {'Positivo' if pred else 'Negativo'}")

if __name__ == "__main__":
    model_path = "final_model_fine_tuned_v2.keras"
    thresholds_path = "optimal_thresholds_partes.json"
    image_path = "../data/predecir/golpe_03.jpg"

    label_to_cls_piezas = {
        1: "Antiniebla delantero derecho",
        2: "Antiniebla delantero izquierdo",
        3: "Capó",
        4: "Cerradura capo",
        5: "Cerradura maletero",
        6: "Cerradura puerta",
        7: "Espejo lateral derecho",
        8: "Espejo lateral izquierdo",
        9: "Faros derecho",
        10: "Faros izquierdo",
        11: "Guardabarros delantero derecho",
        12: "Guardabarros delantero izquierdo",
        13: "Guardabarros trasero derecho",
        14: "Guardabarros trasero izquierdo",
        15: "Luz indicadora delantera derecha",
        16: "Luz indicadora delantera izquierda",
        17: "Luz indicadora trasera derecha",
        18: "Luz indicadora trasera izquierda",
        19: "Luz trasera derecho",
        20: "Luz trasera izquierdo",
        21: "Maletero",
        22: "Manija derecha",
        23: "Manija izquierda",
        24: "Marco de la ventana",
        25: "Marco de las puertas",
        26: "Moldura capó",
        27: "Moldura puerta delantera derecha",
        28: "Moldura puerta delantera izquierda",
        29: "Moldura puerta trasera derecha",
        30: "Moldura puerta trasera izquierda",
        31: "Parabrisas delantero",
        32: "Parabrisas trasero",
        33: "Parachoques delantero",
        34: "Parachoques trasero",
        35: "Puerta delantera derecha",
        36: "Puerta delantera izquierda",
        37: "Puerta trasera derecha",
        38: "Puerta trasera izquierda",
        39: "Rejilla, parrilla",
        40: "Rueda",
        41: "Tapa de combustible",
        42: "Tapa de rueda",
        43: "Techo",
        44: "Techo corredizo",
        45: "Ventana delantera derecha",
        46: "Ventana delantera izquierda",
        47: "Ventana trasera derecha",
        48: "Ventana trasera izquierda",
        49: "Ventanilla delantera derecha",
        50: "Ventanilla delantera izquierda",
        51: "Ventanilla trasera derecha",
        52: "Ventanilla trasera izquierda"
    }

    label_to_cls_danos = {
        1: "Abolladura",
        2: "Deformación",
        3: "Desprendimiento",
        4: "Fractura",
        5: "Rayón",
        6: "Rotura"
    }

    label_to_cls_sugerencias = {
        1: "Reparar",
        2: "Reemplazar"
    }

    model = load_model(model_path)
    mlb_partes, mlb_dannos, mlb_sugerencias = load_mlb_objects()
    thresholds_partes = load_thresholds(thresholds_path)

    predictions = predict_with_thresholds(image_path, model, mlb_partes, mlb_dannos, mlb_sugerencias, thresholds_partes)
    print_predictions(predictions, label_to_cls_piezas, label_to_cls_danos, label_to_cls_sugerencias)
