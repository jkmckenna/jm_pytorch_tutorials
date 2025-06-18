import torch
import torch.nn.functional as F

def get_feature_maps(model, image, layer_name="conv1"):
    """
    Extract feature maps from a specific layer of a CNN model.

    Parameters:
        model (torch.nn.Module): the CNN model
        image (Tensor): input image tensor [C, H, W]
        layer_name (str): name of the convolutional layer (e.g. "conv1")
        device (str or torch.device): device to use

    Returns:
        Tensor: feature maps [num_filters, H, W]
    """
    device = model.device
    model.eval()
    image = image.unsqueeze(0).to(device)  # [1, C, H, W]

    # Register hook to grab the activation
    activations = {}

    def hook_fn(module, input, output):
        activations["feature_map"] = output.detach()

    # Get layer from model
    target_layer = dict(model.named_modules()).get(layer_name, None)
    if target_layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    # Register hook
    hook = target_layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model(image)

    # Cleanup hook
    hook.remove()

    # Return feature maps
    return activations["feature_map"].squeeze().cpu()  # [num_filters, H, W]

def get_conv_filters(conv_layer):
    # Get weights [out_channels, in_channels, H, W]
    weights = conv_layer.weight.data.clone().cpu()
    return weights

def compute_saliency(model, image, label, reduce_channels="max"):
    """
    Compute a 2D saliency map for a single input image with flexible channel reduction.

    Args:
        model (torch.nn.Module): the trained model
        image (torch.Tensor): input image of shape [C, H, W]
        label (int): ground-truth label
        reduce_channels (str): one of {"max", "mean", "norm"}

    Returns:
        torch.Tensor: saliency map of shape [H, W]
    """
    model.eval()
    device = model.device

    image = image.unsqueeze(0).to(device).requires_grad_()  # [1, C, H, W]
    label_tensor = torch.tensor([label], device=device)

    output = model(image)
    loss = F.cross_entropy(output, label_tensor)
    loss.backward()

    grads = image.grad.abs().squeeze(0)  # [C, H, W]

    if reduce_channels == "max":
        saliency_map = grads.max(dim=0)[0]
    elif reduce_channels == "mean":
        saliency_map = grads.mean(dim=0)
    elif reduce_channels == "norm":
        saliency_map = grads.norm(p=2, dim=0)
    else:
        raise ValueError(f"Unknown reduce_channels method: {reduce_channels}")

    return saliency_map.cpu()

def compute_saliency_map(model, image, label=None, device=None):
    model.eval()
    device = device or next(model.parameters()).device
    image = image.to(device).unsqueeze(0).requires_grad_()

    output = model(image)  # logits
    pred_label = output.argmax(dim=1).item() if label is None else label
    loss = F.cross_entropy(output, torch.tensor([pred_label], device=device))
    loss.backward()

    saliency = image.grad.abs().squeeze().detach().cpu()  # [C, H, W]
    return saliency, pred_label
