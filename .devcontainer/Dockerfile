# Use the official miniconda3 image
FROM continuumio/miniconda3:24.4.0-0

# Set the working directory
WORKDIR /workspace

# Copy the environment.yml file
COPY environment.yml .

# Create the conda environment
RUN conda env create -f environment.yml

# Set the environment path
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Make sure the environment is activated when the container starts
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]