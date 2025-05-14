from sklearn.metrics import precision_recall_fscore_support

def print_macro_micro_metrics(y_true, y_pred, class_names, task_name):
    """
    Prints macro and micro averaged precision, recall, and f1-score for multilabel classification.

    Args:
        y_true (np.array): True binary labels, shape (num_samples, num_classes)
        y_pred (np.array): Predicted binary labels, shape (num_samples, num_classes)
        class_names (list): List of class names corresponding to columns in y_true/y_pred
        task_name (str): Name of the task for display
    """
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0)

    print(f"Metrics for {task_name}:")
    print(f"  Macro Precision: {precision_macro:.4f}")
    print(f"  Macro Recall:    {recall_macro:.4f}")
    print(f"  Macro F1-score:  {f1_macro:.4f}")
    print(f"  Micro Precision: {precision_micro:.4f}")
    print(f"  Micro Recall:    {recall_micro:.4f}")
    print(f"  Micro F1-score:  {f1_micro:.4f}")
    print()

# Example usage (run after evaluation_results is available):
# print_macro_micro_metrics(evaluation_results['true_labels']['partes'], evaluation_results['predictions']['partes'], 
#                           [label_to_cls_piezas.get(int(cls), "Otros") for cls in mlb_partes.classes_], "Partes")
# print_macro_micro_metrics(evaluation_results['true_labels']['dannos'], evaluation_results['predictions']['dannos'], 
#                           [label_to_cls_danos.get(int(cls), "Otros") for cls in mlb_dannos.classes_], "Daños")
# print_macro_micro_metrics(evaluation_results['true_labels']['sugerencias'], evaluation_results['predictions']['sugerencias'], 
#                           [label_to_cls_sugerencias.get(int(cls), "Otros") for cls in mlb_sugerencias.classes_], "Sugerencias")
