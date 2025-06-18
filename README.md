# pytorch_tutorials
basic pytorch tutorials and testing

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