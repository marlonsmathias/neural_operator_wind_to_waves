FROM nvcr.io/nvidia/pytorch:23.12-py3
COPY . .

RUN pip install torch-geometric netCDF4
RUN pip install wandb