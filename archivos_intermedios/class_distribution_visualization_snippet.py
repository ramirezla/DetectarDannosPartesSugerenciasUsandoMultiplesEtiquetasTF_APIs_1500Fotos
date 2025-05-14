import matplotlib.pyplot as plt
import pandas as pd

def plot_class_distribution(df, column_name, label_to_cls, task_name, max_classes=20):
    """
    Plots the distribution of samples per class for a multilabel column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column_name (str): Column name with multilabel lists.
        label_to_cls (dict): Mapping from class index to class name.
        task_name (str): Name of the task for plot title.
        max_classes (int): Maximum number of classes to plot.
    """
    # Flatten list of labels
    all_labels = df[column_name].explode()
    # Count occurrences per class
    class_counts = all_labels.value_counts().sort_values(ascending=False)
    # Limit to max_classes
    class_counts = class_counts.head(max_classes)
    # Map class indices to names
    class_names = [label_to_cls.get(int(cls), "Otros") for cls in class_counts.index]

    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_counts.values)
    plt.xticks(rotation=90)
    plt.title(f"Distribución de muestras por clase - {task_name}")
    plt.ylabel("Número de muestras")
    plt.xlabel("Clases")
    plt.tight_layout()
    plt.show()

# Example usage (run in notebook):
# plot_class_distribution(train_df, 'partes', label_to_cls_piezas, 'Partes')
# plot_class_distribution(train_df, 'dannos', label_to_cls_danos, 'Daños')
# plot_class_distribution(train_df, 'sugerencias', label_to_cls_sugerencias, 'Sugerencias')
