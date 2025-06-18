import matplotlib.pyplot as plt
import numpy as np
import math
import torch

def plot_feature_maps(input_image, maps, title="Conv Layer Activations"):
    """
    Plot the input image (raw) and a set of feature maps (in viridis) from a CNN layer.

    Parameters:
        input_image (Tensor): shape [C, H, W] or [H, W]
        maps (Tensor): shape [N, H, W], the feature maps
        title (str): optional plot title
    """
    num_maps = maps.shape[0]
    num_cols = 8
    total_maps = num_maps + 1  # include input image
    num_rows = (total_maps + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 1.5), constrained_layout=True)
    axes = axes.flatten()

    # Normalize feature maps for consistent color scaling
    vmin = maps.min().item()
    vmax = maps.max().item()
    img_handle = None  # to save one imshow handle for the colorbar

    # Plot the input image (raw, grayscale or RGB)
    ax = axes[0]
    img = input_image.cpu()
    if img.ndim == 3 and img.shape[0] != 1:  # [C, H, W]
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)  # Let matplotlib infer RGB or grayscale
    else:
        img = img.squeeze().numpy()
        ax.imshow(img, cmap='gray')
    ax.set_title("Input Image")
    ax.axis('off')

    # Plot the feature maps
    for i in range(num_maps):
        ax = axes[i + 1]
        im = ax.imshow(maps[i].cpu(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        if img_handle is None:
            img_handle = im  # save handle for colorbar

    # Hide any unused axes
    for j in range(num_maps + 1, len(axes)):
        axes[j].axis('off')

    # Add shared colorbar to the right
    if img_handle:
        cbar = fig.colorbar(img_handle, ax=axes, location='right', shrink=0.8, pad=0.02)
        cbar.set_label("Activation Intensity")

    plt.suptitle(title)
    plt.show()

def plot_dense_activation(activations, title=None, softmax=True):
    """
    Plot 1D dense activations as a bar chart.
    """

    if softmax:
        activations = torch.softmax(activations, dim=0).cpu().numpy()
    else:
        activations = activations.cpu().numpy()
    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(len(activations)), activations)
    plt.title(title or "Dense Layer Activations")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.show()

def plot_conv_filters(weights, title="Conv Filters"):
    """
    Plot convolutional filters with a shared intensity scale bar (colorbar).

    Parameters:
        weights (Tensor or ndarray): shape [out_channels, in_channels, H, W]
        title (str): plot title
    """
    num_filters = weights.shape[0]
    num_cols = 8
    num_rows = (num_filters + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows), constrained_layout=True)
    axes = axes.flatten()

    # Collect all weights for consistent vmin/vmax
    all_weights = weights[:, 0]  # assuming single input channel
    vmin = all_weights.min().item()
    vmax = all_weights.max().item()

    img_handle = None
    for i in range(num_filters):
        ax = axes[i]
        w = weights[i, 0]  # shape: [H, W]
        img_handle = ax.imshow(w, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')

    for j in range(num_filters, len(axes)):
        axes[j].axis('off')

    # Add colorbar
    cbar = fig.colorbar(img_handle, ax=axes, location='right', shrink=0.8, pad=0.02)
    cbar.set_label("Filter Weight (Intensity)")

    plt.suptitle(title)
    plt.show()

def show_saliency_map(image, saliency_map, title=None):
    """
    Plot the input image and corresponding saliency map side by side.

    Args:
        image (torch.Tensor): input image, shape [C, H, W]
        saliency_map (torch.Tensor): saliency map, shape [H, W]
        title (str): optional plot title
    """
    image = image.detach().cpu()
    saliency_map = saliency_map.detach().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(4, 2))

    # Convert image to (H, W, C) if needed
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        img_disp = image.permute(1, 2, 0).numpy()
        if image.shape[0] == 1:  # grayscale
            img_disp = img_disp.squeeze(-1)
        axes[0].imshow(img_disp, cmap='gray' if image.shape[0] == 1 else None)
    else:
        axes[0].imshow(image.numpy(), cmap='gray')

    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title("Saliency Map")
    axes[1].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_saliency_map(image, saliency, title=None):
    """
    image: torch.Tensor [C, H, W]
    saliency: torch.Tensor [C, H, W] or [H, W]
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()
    saliency_np = saliency.mean(dim=0).numpy() if saliency.ndim == 3 else saliency.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[1].imshow(saliency_np, cmap="hot")
    axes[1].set_title("Saliency Map")
    if title:
        fig.suptitle(title)
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