# pytorch_tutorials
basic pytorch tutorials and testing


# Make an environment from the environment.yaml file
conda env create -f environment.yml

# Activate the environment
conda activate pytorch-tutorials

# Register the kernel after creating the first time
python -m ipykernel install --user --name pytorch-tutorials --display-name "Python (PyTorch Tutorials)"

# If updating the environment.yaml file later, run:
conda env update -f environment.yml --prune