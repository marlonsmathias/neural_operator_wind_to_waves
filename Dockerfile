FROM nvcr.io/nvidia/pytorch:23.12-py3
COPY . .
RUN rm -r .git

RUN pip install torch-geometric netCDF4 parfor
RUN pip install wandb