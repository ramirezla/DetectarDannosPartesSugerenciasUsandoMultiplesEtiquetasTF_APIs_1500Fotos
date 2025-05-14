# Cell 1: Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns

# Cell 2: Function to plot confusion matrices
def plot_confusion_matrices(y_true, y_pred, class_names, task_name, max_classes=10):
    """
    Plots confusion matrices for multilabel classification task.

    Args:
        y_true (np.array): True binary labels, shape (num_samples, num_classes)
        y_pred (np.array): Predicted binary labels, shape (num_samples, num_classes)
        class_names (list): List of class names corresponding to columns in y_true/y_pred
        task_name (str): Name of the task (for plot titles)
        max_classes (int): Maximum number of classes to plot (to avoid too many plots)
    """
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)
    classes_to_plot = min(num_classes, max_classes)

    plt.figure(figsize=(15, 4 * classes_to_plot))
    for i in range(classes_to_plot):
        cm = mcm[i]
        tn, fp, fn, tp = cm.ravel()
        cm_matrix = np.array([[tp, fn],
                              [fp, tn]])
        ax = plt.subplot(classes_to_plot, 1, i + 1)
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Actual Positive', 'Actual Negative'],
                    yticklabels=['Predicted Positive', 'Predicted Negative'])
        ax.set_title(f'{task_name} - Confusion Matrix for Class: {class_names[i]}')
        ax.set_ylabel('Predicted label')
        ax.set_xlabel('True label')
    plt.tight_layout()
    plt.show()

# Cell 3: Function to generate confusion matrices by task
def generate_confusion_matrices_by_task(evaluation_results, mlb_partes, mlb_dannos, mlb_sugerencias,
                                        label_to_cls_piezas, label_to_cls_danos, label_to_cls_sugerencias):
    """
    Generate and plot confusion matrices for parts, damages, and suggestions tasks.

    Args:
        evaluation_results (dict): Dictionary containing 'true_labels' and 'predictions' for each task.
        mlb_partes, mlb_dannos, mlb_sugerencias: MultiLabelBinarizer objects for each task.
        label_to_cls_piezas, label_to_cls_danos, label_to_cls_sugerencias: Dicts mapping class indices to names.
    """
    # Extract true and predicted labels
    y_true_partes = evaluation_results['true_labels']['partes']
    y_pred_partes = evaluation_results['predictions']['partes']

    y_true_dannos = evaluation_results['true_labels']['dannos']
    y_pred_dannos = evaluation_results['predictions']['dannos']

    y_true_sugerencias = evaluation_results['true_labels']['sugerencias']
    y_pred_sugerencias = evaluation_results['predictions']['sugerencias']

    # Map class indices to names
    partes_names = [label_to_cls_piezas.get(int(cls), f"Clase_{cls}") for cls in mlb_partes.classes_]
    dannos_names = [label_to_cls_danos.get(int(cls), f"Clase_{cls}") for cls in mlb_dannos.classes_]
    sugerencias_names = [label_to_cls_sugerencias.get(int(cls), f"Clase_{cls}") for cls in mlb_sugerencias.classes_]

    # Plot confusion matrices for each task
    print("Plotting confusion matrices for Partes...")
    plot_confusion_matrices(y_true_partes, y_pred_partes, partes_names, "Partes")

    print("Plotting confusion matrices for Daños...")
    plot_confusion_matrices(y_true_dannos, y_pred_dannos, dannos_names, "Daños")

    print("Plotting confusion matrices for Sugerencias...")
    plot_confusion_matrices(y_true_sugerencias, y_pred_sugerencias, sugerencias_names, "Sugerencias")

# Cell 4: Example usage (run after evaluation_results is available)
# generate_confusion_matrices_by_task(
#     evaluation_results,
#     mlb_partes, mlb_dannos, mlb_sugerencias,
#     label_to_cls_piezas, label_to_cls_danos, label_to_cls_sugerencias
# )
