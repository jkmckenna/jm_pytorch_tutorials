import torch
from torch import nn
from sklearn.base import BaseEstimator

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

