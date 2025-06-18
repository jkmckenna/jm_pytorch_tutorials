import torch
from torch import nn
import time

def train_torch_model(model, train_loader, epochs=5,
                optimizer_class=torch.optim.Adam,
                learning_rate=0.001,
                criterion=nn.CrossEntropyLoss()):
    """
    Train a PyTorch model with optional custom optimizer, learning rate, and loss function.

    Parameters:
        model (nn.Module): the model to train
        train_loader (DataLoader): training data loader
        epochs (int): number of training epochs
        optimizer_class (torch.optim.Optimizer): optimizer class (not instance)
        learning_rate (float): learning rate
        criterion (nn.Module): loss function
    """
    device = model.device

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.to(device)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    total_train_time = 0.0

    print(f"Training model: {model.name}")
    print(f"Trainable parameters: {num_params:,}")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        total_train_time += epoch_time
        print(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss:.4f} | Time = {epoch_time:.2f}s")

    print(f"Total training time: {total_train_time:.2f}s")
    model.train_time = total_train_time
