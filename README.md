# jm_pytorch_tutorials
basic pytorch tutorials

# Purpose
A crash course introduction for the Tjian/Darzacq Lab into computer vision and representation learning in PyTorch. A basic (MNIST training/eval and feature map intro), intermediate (CIFAR10 training/eval and feature maps), and advanced introduction (Various pretrained Imagenet1000 models inference on new images, feature maps, attention maps) are provided.

# Clone repo from source
```shell
git clone https://github.com/jkmckenna/jm_pytorch_tutorials
```

# Enter repo
```shell
cd jm_pytorch_tutorials
```

# Make an environment from the environment.yaml file
```shell
conda env create -f environment.yml
```

# Activate the environment
```shell
conda activate pytorch-tutorials
```

# Register the kernel after creating the first time
```shell
python -m ipykernel install --user --name pytorch-tutorials --display-name "Python (PyTorch Tutorials)"
```

# Install the jm_pytorch_tutorials python package in editable mode
```shell
pip install -e .
```

# If updating the environment.yaml file later, run:
```shell
conda env update -f environment.yml --prune
```
