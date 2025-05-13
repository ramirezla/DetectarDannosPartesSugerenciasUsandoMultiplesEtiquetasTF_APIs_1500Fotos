import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os

def evaluate_model_with_thresholds(model, generator, mlb_partes, mlb_dannos, mlb_sugerencias, thresholds):
    results = model.evaluate(generator, verbose=1)
    print(f"Total loss: {results[0]}")
    print(f"Partes loss: {results[1]} - Partes accuracy: {results[4]}")
    print(f"Dannos loss: {results[2]} - Dannos accuracy: {results[5]}")
    print(f"Sugerencias loss: {results[3]} - Sugerencias accuracy: {results[6]}")

    predictions = model.predict(generator, verbose=1)

    y_true_partes = generator.mlb_partes.transform(generator.df['partes'])
    y_true_dannos = generator.mlb_dannos.transform(generator.df['dannos'])
    y_true_sugerencias = generator.mlb_sugerencias.transform(generator.df['sugerencias'])

    y_pred_partes = (predictions[0] > thresholds['partes']).astype(int)
    y_pred_dannos = (predictions[1] > thresholds['dannos']).astype(int)
    y_pred_sugerencias = (predictions[2] > thresholds['sugerencias']).astype(int)

    print("\nClassification Report for Partes:")
    print(classification_report(y_true_partes, y_pred_partes, zero_division=0))

    print("\nClassification Report for Dannos:")
    print(classification_report(y_true_dannos, y_pred_dannos, zero_division=0))

    print("\nClassification Report for Sugerencias:")
    print(classification_report(y_true_sugerencias, y_pred_sugerencias, zero_division=0))

if __name__ == "__main__":
    import sys
    import os
    import pickle

    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    @tf.keras.utils.register_keras_serializable()
    def swish(x):
        return x * tf.keras.backend.sigmoid(x)

    model = load_model(model_path, compile=False, custom_objects={'swish': swish})

    with open("mlb_partes.pkl", "rb") as f:
        mlb_partes = pickle.load(f)
    with open("mlb_dannos.pkl", "rb") as f:
        mlb_dannos = pickle.load(f)
    with open("mlb_sugerencias.pkl", "rb") as f:
        mlb_sugerencias = pickle.load(f)

    val_csv = os.path.join('data/fotos_siniestros/val', 'val_set.csv')
    val_df = pd.read_csv(val_csv, sep='|')
    for col in ['dannos', 'partes', 'sugerencias']:
        val_df[col] = val_df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    # BalancedMultiLabelDataGenerator class should be defined or imported here

    val_generator = BalancedMultiLabelDataGenerator(
        val_df, '../data/fotos_siniestros/', mlb_partes, mlb_dannos, mlb_sugerencias,
        batch_size=32, img_size=(224,224), shuffle=False
    )

    with open('optimized_thresholds.json', 'r') as f:
        thresholds = json.load(f)

    evaluate_model_with_thresholds(model, val_generator, mlb_partes, mlb_dannos, mlb_sugerencias, thresholds)
