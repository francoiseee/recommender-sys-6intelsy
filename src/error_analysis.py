import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def analyze_failures(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    failure_cases = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                failure_cases.append((classes[i], classes[j], cm[i, j]))
    return failure_cases


def subgroup_analysis(y_true, y_pred, subgroups, classes):
    subgroup_results = {}
    for group in subgroups:
        indices = np.where(subgroups == group)[0]
        subgroup_y_true = y_true[indices]
        subgroup_y_pred = y_pred[indices]
        subgroup_results[group] = confusion_matrix(subgroup_y_true, subgroup_y_pred)
    return subgroup_results


# Example Usage
# y_true = np.array([...])  # True labels
# y_pred = np.array([...])  # Predicted labels
# classes = ['Class1', 'Class2', 'Class3']  # List of class names
# plot_confusion_matrix(y_true, y_pred, classes)
# failure_cases = analyze_failures(y_true, y_pred, classes)
# print(f'Failure cases: {failure_cases}')  
# subgroups = np.array([...])  # Subgroup labels corresponding to y_true
# subgroup_results = subgroup_analysis(y_true, y_pred, subgroups, classes)