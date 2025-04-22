#!/bin/bash

# Check if mpirun exists
if ! command -v mpirun &> /dev/null; then
    echo "[info] OpenMPI not found, installing..."
    sudo apt-get update
    sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev
else
    echo "[info] OpenMPI already installed"
fi

# Install cuda-python, mpi4py, triton
echo "[info] Installing cuda-python and mpi4py (could be long)"
pip install cuda-python
pip install mpi4py
pip install triton

echo "[info] Installation completed"