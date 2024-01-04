FROM nvcr.io/nvidia/pytorch:22.11-py3
COPY . .

RUN pip install torch-geometric parfor
RUN pip install wandb