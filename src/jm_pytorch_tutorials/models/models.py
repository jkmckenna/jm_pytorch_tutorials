import torch
from torch import nn
from sklearn.base import BaseEstimator

from torchvision.models import resnet18, ResNet18_Weights
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


import time

from ..utils import get_device
from ..test import evaluate_sklearn_model

class SklearnModelWrapper:
    def __init__(self, model: BaseEstimator, name=None, class_names=None, num_classes=10):
        """
        Wrap a scikit-learn model for standardized training and evaluation.

        Parameters:
            model (BaseEstimator): an sklearn classifier (e.g., SVC, RF, NB)
            name (str): model name
            class_names (list): optional list of class labels
            num_classes (int): number of classes
        """
        self.model = model
        self.name = name or type(model).__name__
        self.class_names = class_names
        self.num_classes = num_classes
        self.eval_metrics = None
        self.train_time = None

    def fit(self, X, y):
        print(f"Training {self.name}")
        start = time.time()
        self.model.fit(X, y)
        end = time.time()
        self.train_time = end - start
        print(f"Training time: {self.train_time:.2f}s")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        print(f"Evaluating {self.name}")
        self.eval_metrics = evaluate_sklearn_model(
            self,
            X_test=X,
            y_test=y,
            class_names=self.class_names,
            num_classes=self.num_classes
        )
        return self.eval_metrics

class MLP(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), hidden_dims=[512], num_classes=10):
        super().__init__()
        self.device = get_device()
        self.name = "MLP"

        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        layers = []

        # First layer
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        # Final output layer
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)

class CNN(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), conv_channels=[32, 64], fc_dims=[128], num_classes=10):
        super().__init__()
        self.device = get_device()
        self.name = "CNN"

        channels, height, width = input_shape

        layers = []
        in_channels = channels

        # Convolutional layers
        for out_channels in conv_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Halve H and W
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Compute flattened size after conv + pooling
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy)
            self.flattened_size = conv_out.view(1, -1).shape[1]

        # Fully connected layers
        fc_layers = []
        in_dim = self.flattened_size
        for h in fc_dims:
            fc_layers.append(nn.Linear(in_dim, h))
            fc_layers.append(nn.ReLU())
            in_dim = h

        fc_layers.append(nn.Linear(in_dim, num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=None, pretrained=True):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.device = get_device()
        if num_classes is not None:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        self.transform = ResNet18_Weights.DEFAULT.transforms() if pretrained else None
        self.layers_to_track = ['conv1', 'relu', 'layer1.0.conv1', 'layer1.0.relu']

    def forward(self, x):
        return self.model(x)

    def preprocess(self, image):
        return self.transform(image) if self.transform else image


class ViTClassifier(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=None, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.device = get_device()
        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)
        self.layers_to_track = ['blocks.0.attn', 'blocks.1.attn', 'blocks.2.attn', 'blocks.3.attn']

    def forward(self, x):
        return self.model(x)

    def preprocess(self, image):
        return self.transform(image)


class ConvNeXtClassifier(nn.Module):
    def __init__(self, model_name='convnext_base', num_classes=None, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.device = get_device()
        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)
        self.layers_to_track = ['stem.0', 'stages.0.blocks.0.conv_dw', 'stages.0.blocks.0.mlp.act']

    def forward(self, x):
        return self.model(x)

    def preprocess(self, image):
        return self.transform(image)