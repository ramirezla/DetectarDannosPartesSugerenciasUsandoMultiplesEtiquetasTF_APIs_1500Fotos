import os
import glob
from predict_with_custom_thresholds import load_model, load_mlb_objects, load_thresholds, predict_with_thresholds, print_predictions

def main():
    model_path = "final_model_fine_tuned_v2.keras"
    thresholds_path = "optimal_thresholds_partes.json"
    image_folder = "../data/predecir/"

    model = load_model(model_path)
    mlb_partes, mlb_dannos, mlb_sugerencias = load_mlb_objects()
    thresholds_partes = load_thresholds(thresholds_path)

    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    if not image_paths:
        print(f"No se encontraron imágenes en la carpeta {image_folder}")
        return

    for img_path in image_paths:
        print(f"\nPredicciones para la imagen: {os.path.basename(img_path)}")
        predictions = predict_with_thresholds(img_path, model, mlb_partes, mlb_dannos, mlb_sugerencias, thresholds_partes)
        # Diccionarios de mapeo con nombres reales
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
        print_predictions(predictions, label_to_cls_piezas, label_to_cls_danos, label_to_cls_sugerencias)

if __name__ == "__main__":
    main()
