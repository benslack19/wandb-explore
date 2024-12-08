# Use Ubuntu as the base image
FROM ubuntu:latest

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Add conda to path
ENV PATH=$CONDA_DIR/bin:$PATH

# Create a conda environment from the YAML file
COPY environments/wandb.yml /tmp/wandb.yml
RUN conda env create -f /tmp/wandb.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "wandb", "/bin/bash", "-c"]

# Set the default command to activate the conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "wandb", "/bin/bash"]
