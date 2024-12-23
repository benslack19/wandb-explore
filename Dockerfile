# Edit of miniforge3 for mamba Dockerfile
# https://github.com/conda-forge/miniforge-images/blob/master/ubuntu/Dockerfile

FROM ubuntu:focal-20241011

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=24.9.2-0
ARG TARGETPLATFORM

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

# 1. Install just enough for conda to work
# 2. Keep $HOME clean (no .wget-hsts file), since HSTS isn't useful in this context
# 3. Install miniforge from GitHub releases
# 4. Apply some cleanup tips from https://jcrist.github.io/conda-docker-tips.html
#    Particularly, we remove pyc and a files. The default install has no js, we can skip that
# 5. Activate base by default when running as any *non-root* user as well
#    Good security practice requires running most workloads as non-root
#    This makes sure any non-root users created also have base activated
#    for their interactive shells.
# 6. Activate base by default when running as root as well
#    The root user is already created, so won't pick up changes to /etc/skel
RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        git \
        tini \
        > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc && \
    if [ ${MINIFORGE_NAME} = "Mambaforge"* ]; then \
        echo '. ${CONDA_DIR}/etc/.mambaforge_deprecation.sh' >> /etc/skel/.bashrc && \
        echo '. ${CONDA_DIR}/etc/.mambaforge_deprecation.sh' >> ~/.bashrc; \
    fi

ENTRYPOINT ["tini", "--"]

# # Create a mamba environment from the YAML file
# COPY environments/wandb.yml /tmp/wandb.yml
# RUN mamba env create -f /tmp/wandb.yml

# # Make RUN commands use the new environment
# SHELL ["mamba", "run", "-n", "wandb", "/bin/bash", "-c"]

# # Set the default command to activate the conda environment
# CMD ["mamba", "run", "--no-capture-output", "-n", "wandb", "/bin/bash"]

# Add packages based on conda/mamba
# Copy the YAML file
COPY environments/wandb.yml /tmp/wandb.yml

# Install packages from the YAML file into the base environment
RUN conda env update -n base -f /tmp/wandb.yml && \
    conda clean --all --yes

# Set the default command
CMD ["/bin/bash"]

# Run as non-root user for security reasons -------
# Create a non-root user
RUN useradd -m -s /bin/bash nonrootuser

# Set the working directory to the user's home
WORKDIR /home/nonrootuser

# Change ownership of the conda installation
RUN chown -R nonrootuser:nonrootuser $CONDA_DIR

# Switch to the non-root user
USER nonrootuser
