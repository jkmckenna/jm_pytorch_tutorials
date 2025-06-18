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

def compute_saliency(model, image, label):
    model.eval()
    image = image.unsqueeze(0).to(model.device).requires_grad_()
    output = model(image)
    loss = F.cross_entropy(output, torch.tensor([label]).to(model.device))
    loss.backward()

    saliency = image.grad.abs().squeeze().cpu()
    return saliency
