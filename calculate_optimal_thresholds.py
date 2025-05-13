import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.metrics import precision_recall_curve, f1_score

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

def load_test_data(test_csv_path):
    df = pd.read_csv(test_csv_path, sep='|')
    import json
    for col in ['dannos', 'partes', 'sugerencias']:
        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df

def get_optimal_thresholds(y_true, y_scores):
    n_classes = y_true.shape[1]
    optimal_thresholds = []
    for i in range(n_classes):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_scores[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        if best_idx >= len(thresholds):
            # best_idx can be len(thresholds) when recall=1 and precision=0 at last point
            best_threshold = 1.0
        else:
            best_threshold = thresholds[best_idx]
        optimal_thresholds.append(best_threshold)
    return np.array(optimal_thresholds)

def save_thresholds(thresholds, class_names, filename):
    thresholds_dict = {class_names[i]: float(thresholds[i]) for i in range(len(class_names))}
    with open(filename, 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f"Umbrales óptimos guardados en {filename}")

if __name__ == "__main__":
    model_path = "final_model_fine_tuned_v2.keras"
    test_csv_path = "data/fotos_siniestros/test/test_set.csv"

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

    model = load_model(model_path)
    mlb_partes, mlb_dannos, mlb_sugerencias = load_mlb_objects()
    test_df = load_test_data(test_csv_path)

    y_true_partes = mlb_partes.transform(test_df['partes'])

    # Preparar datos de imágenes para predicción
    IMG_DIR = '../data/fotos_siniestros/'
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    X_test = []
    for img_name in test_df['Imagen']:
        img_path = os.path.join(IMG_DIR, img_name)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        X_test.append(img_array)
    X_test = np.array(X_test)

    predictions = model.predict(X_test, batch_size=32, verbose=1)
    y_scores_partes = predictions[0]

    optimal_thresholds_partes = get_optimal_thresholds(y_true_partes, y_scores_partes)
    class_names_partes = [label_to_cls_piezas[i+1] for i in range(len(optimal_thresholds_partes))]

    save_thresholds(optimal_thresholds_partes, class_names_partes, "optimal_thresholds_partes.json")
