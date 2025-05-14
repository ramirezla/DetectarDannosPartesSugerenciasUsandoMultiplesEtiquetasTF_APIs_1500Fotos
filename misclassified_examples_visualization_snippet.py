import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_misclassified_examples(df, y_true, y_pred, label_to_cls, task_name, mlb, max_examples=5, base_image_path='../data/fotos_siniestros/'):
    """
    Visualizes misclassified examples for a multilabel classification task.

    Args:
        df (pd.DataFrame): DataFrame containing image file names and labels.
        y_true (np.array): True binary labels, shape (num_samples, num_classes).
        y_pred (np.array): Predicted binary labels, shape (num_samples, num_classes).
        label_to_cls (dict): Mapping from class index to class name.
        task_name (str): Name of the task for display.
        mlb (MultiLabelBinarizer): The fitted binarizer for inverse transform.
        max_examples (int): Maximum number of misclassified examples to display.
        base_image_path (str): Base directory path where images are stored.
    """
    # Find misclassified indices (where any label differs)
    misclassified_indices = np.where(np.any(y_true != y_pred, axis=1))[0]

    if len(misclassified_indices) == 0:
        print(f"No hay ejemplos mal clasificados para la tarea {task_name}.")
        return

    print(f"Mostrando hasta {max_examples} ejemplos mal clasificados para la tarea {task_name} (total: {len(misclassified_indices)})")

    for idx in misclassified_indices[:max_examples]:
        img_name = df.iloc[idx]['Imagen'] if 'Imagen' in df.columns else None
        if img_name is None:
            print(f"Nombre de imagen no encontrado para índice {idx}.")
            print(f"Información adicional del dataframe para índice {idx}:")
            print(df.iloc[idx])
            continue

        img_path = os.path.join(base_image_path, img_name)
        if not os.path.exists(img_path):
            print(f"Imagen no encontrada para índice {idx}. Ruta esperada: {img_path}")
            print(f"Información adicional del dataframe para índice {idx}:")
            print(df.iloc[idx])
            continue

        img = plt.imread(img_path)
        true_labels = mlb.inverse_transform(y_true[idx].reshape(1, -1))[0]
        pred_labels = mlb.inverse_transform(y_pred[idx].reshape(1, -1))[0]

        true_label_names = [label_to_cls.get(int(cls), "Otros") for cls in true_labels]
        pred_label_names = [label_to_cls.get(int(cls), "Otros") for cls in pred_labels]

        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{task_name} - Imagen: {img_name} - Índice {idx}\nVerdadero: {', '.join(true_label_names)}\nPredicho: {', '.join(pred_label_names)}", fontsize=10)
        plt.show()

# Ejemplo de uso (en notebook):
# visualize_misclassified_examples(train_df, evaluation_results['true_labels']['partes'], evaluation_results['predictions']['partes'], 
#                                  label_to_cls_piezas, "Partes", mlb_partes, max_examples=5)
