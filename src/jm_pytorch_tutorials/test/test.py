from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
import torch
from sklearn.preprocessing import label_binarize
import numpy as np
import time

def evaluate_sklearn_model(clf, X_test, y_test, class_names=None, num_classes=10, save_to_model=True):
    """
    Evaluate a trained scikit-learn classifier.

    Parameters:
        clf: a wrapped sklearn model with clf.model already trained
        X_test: input features
        y_test: ground truth labels
        class_names: optional list of class names
        num_classes: total number of classes
        name: model label for plotting/logging

    Returns:
        dict containing metrics and curves
    """
    name = clf.name

    print(f"Evaluating model: {clf.name}")
    # Binarize true labels
    y_true_bin = label_binarize(y_test, classes=np.arange(num_classes))

    # Predict probabilities
    start = time.time()
    y_score = clf.predict_proba(X_test)
    end = time.time()

    # Hard predictions
    y_pred = y_score.argmax(axis=1)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # AUC scores
    roc_auc = roc_auc_score(y_true_bin, y_score, average=None)
    pr_auc = average_precision_score(y_true_bin, y_score, average=None)
    macro_auroc = roc_auc_score(y_true_bin, y_score, average="macro")
    macro_auprc = average_precision_score(y_true_bin, y_score, average="macro")

    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # Per-class ROC and PR curves
    roc_curves = {}
    pr_curves = {}
    for i in range(num_classes):
        label = class_names[i] if class_names else str(i)
        fpr, tpr, roc_thresh = roc_curve(y_true_bin[:, i], y_score[:, i])
        precision, recall, pr_thresh = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        roc_curves[label] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresh.tolist(),
        }
        pr_curves[label] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresh.tolist(),
        }

    # Flattened macro-average ROC and PR curves
    fpr_macro, tpr_macro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    precision_macro, recall_macro, _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())

    metrics = {
        "label": name,
        "accuracy": acc,
        "train_time": clf.train_time,
        "inference_time": end - start,
        "auroc": macro_auroc,
        "auprc": macro_auprc,
        "auroc_per_class": dict(zip(class_names or range(num_classes), roc_auc)),
        "auprc_per_class": dict(zip(class_names or range(num_classes), pr_auc)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "macro_roc_curve": {
            "fpr": fpr_macro.tolist(),
            "tpr": tpr_macro.tolist()
        },
        "macro_pr_curve": {
            "precision": precision_macro.tolist(),
            "recall": recall_macro.tolist()
        }
    }

    if save_to_model:
        clf.eval_metrics = metrics

    return metrics


def evaluate_torch_model(model, test_loader, class_names=None, save_to_model=True):
    device = model.device
    model.eval()
    model.to(device)

    print(f"Evaluating model: {model.name}")

    all_preds = []
    all_probs = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    elapsed = time.time() - start_time

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    num_classes = all_probs.shape[1]
    y_true_bin = label_binarize(all_labels, classes=np.arange(num_classes))

    accuracy = 100 * np.mean(all_preds == all_labels)
    roc_auc = roc_auc_score(y_true_bin, all_probs, average=None)
    pr_auc = average_precision_score(y_true_bin, all_probs, average=None)
    macro_roc_auc = roc_auc_score(y_true_bin, all_probs, average="macro")
    macro_pr_auc = average_precision_score(y_true_bin, all_probs, average="macro")
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    # ROC and PR curves
    roc_curves = {}
    pr_curves = {}
    for i in range(num_classes):
        fpr, tpr, roc_thresh = roc_curve(y_true_bin[:, i], all_probs[:, i])
        precision, recall, pr_thresh = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
        label = class_names[i] if class_names else str(i)
        roc_curves[label] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresh.tolist(),
        }
        pr_curves[label] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresh.tolist(),
        }

    print(f"\nEvaluation Accuracy: {accuracy:.2f}% | Time: {elapsed:.2f}s")
    print("Per-Class ROC AUC and PR AUC:")
    for i in range(num_classes):
        name = class_names[i] if class_names else f"class_{i}"
        print(f"  {name}: ROC AUC = {roc_auc[i]:.3f}, PR AUC = {pr_auc[i]:.3f}")
    print(f"\nMacro ROC AUC: {macro_roc_auc:.3f} | Macro PR AUC: {macro_pr_auc:.3f}")

    # Flattened macro-average ROC and PR curves
    fpr_macro, tpr_macro, _ = roc_curve(y_true_bin.ravel(), all_probs.ravel())
    precision_macro, recall_macro, _ = precision_recall_curve(y_true_bin.ravel(), all_probs.ravel())

    # Save metrics
    metrics = {
        "label": model.name,
        "accuracy": accuracy,
        "auroc_per_class": dict(zip(class_names or range(num_classes), roc_auc)),
        "auprc_per_class": dict(zip(class_names or range(num_classes), pr_auc)),
        "auroc": macro_roc_auc,
        "auprc": macro_pr_auc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "train_time": model.train_time,
        "inference_time": elapsed,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "macro_roc_curve": {
            "fpr": fpr_macro.tolist(),
            "tpr": tpr_macro.tolist()
        },
        "macro_pr_curve": {
            "precision": precision_macro.tolist(),
            "recall": recall_macro.tolist()
        }
    }

    if save_to_model:
        model.eval_metrics = metrics

    return metrics
