FROM nvcr.io/nvidia/pytorch:22.11-py3
COPY . .

RUN pip install torch-geometric parfor netCDF4
RUN pip install wandb