import matplotlib.pyplot as plt
import numpy as np
import math

def plot_feature_maps(input_image, maps, title="Conv1 Activations"):
    num_maps = maps.shape[0]
    num_cols = 8
    total_maps = num_maps + 1  # +1 for input image
    num_rows = (total_maps + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    axes = axes.flatten()

    # Plot the input image first
    axes[0].imshow(input_image.squeeze(), cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Plot activation maps
    for i in range(num_maps):
        ax = axes[i + 1]
        ax.imshow(maps[i], cmap='viridis')
        ax.axis('off')

    # Hide unused axes
    for j in range(num_maps + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_conv_filters(weights, title="Conv Filters"):
    num_filters = weights.shape[0]
    num_cols = 8
    num_rows = (num_filters + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    for i in range(num_filters):
        ax = axes[i // num_cols, i % num_cols]
        w = weights[i, 0]  # Assuming single input channel (e.g. grayscale)
        ax.imshow(w, cmap='gray')
        ax.axis('off')
    for j in range(i + 1, num_rows * num_cols):
        axes[j // num_cols, j % num_cols].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def show_saliency_map(image, saliency):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title("Original Image")
    axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title("Saliency Map")
    plt.tight_layout()
    plt.show()

def plot_classwise_maps(classifier, model_type="nb"):
    if model_type == "svm":
        weights = classifier.coef_
        cmap = "bwr"
    elif model_type == "nb":
        weights = classifier.theta_
        cmap = "viridis"
    else:
        raise ValueError("model_type must be 'svm' or 'nb'")

    n_classes, n_features = weights.shape
    side = int(np.sqrt(n_features))

    if side * side != n_features:
        raise ValueError(
            f"Cannot reshape {n_features} features into square (e.g. 28x28). "
            f"Use rectangular shape or specify reshape manually."
        )

    maps = weights.reshape(n_classes, side, side)

    # Set color scale
    if model_type == "svm":
        vmax = np.abs(maps).max()
        vmin = -vmax
    else:
        vmin, vmax = maps.min(), maps.max()

    # Dynamic grid layout
    n_cols = 5
    n_rows = math.ceil(n_classes / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows), constrained_layout=True)
    axes = axes.flatten()

    for i in range(n_classes):
        ax = axes[i]
        im = ax.imshow(maps[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Class {i}")
        ax.axis("off")

    # Hide extra unused axes
    for j in range(n_classes, len(axes)):
        axes[j].axis("off")

    fig.colorbar(im, ax=axes, location='right', shrink=0.8, pad=0.02, label="Weight / Log-Prob")
    plt.suptitle(f"{model_type.upper()} Class-Specific Feature Importance", fontsize=16)
    plt.show()

def plot_roc_pr_curves(results):
    plt.figure(figsize=(14, 6))

    # ROC
    plt.subplot(1, 2, 1)
    for res in results:
        plt.plot(res['macro_roc_curve']['fpr'], res['macro_roc_curve']['tpr'], label=f"{res['label']} (AUROC = {res['auroc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    # PRC
    plt.subplot(1, 2, 2)
    for res in results:
        plt.plot(res['macro_pr_curve']['recall'], res['macro_pr_curve']['precision'], label=f"{res['label']} (AUPRC = {res['auprc']:.4f})")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()
    plt.show()