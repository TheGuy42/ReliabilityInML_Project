# build: docker build -t rel_image .
# Start new container: docker run --name reliability-container -v "$(pwd):/workspace" -it rel_image


# Use the Conda Forge Miniforge3 image as the base image
FROM condaforge/miniforge3:latest

# Set the working directory inside the container
WORKDIR /root/workspace

ARG ENV_NAME=torch
ARG ACT_ENV=torch
ARG PYTHON_VER=3.10

# Install common software development tools and libraries
# RUN apt-get update && apt-get install -y \
#     git \
#     vim \
#     curl \
#     wget \
#     build-essential \
#     sudo

RUN conda init

# Install Conda and create a new environment
RUN conda create -n ${ENV_NAME} --yes python=PYTHON_VER && \
    echo "conda activate ${ENV_NAME}" >> ~/.bashrc

# Set environment path to use the new Conda environment
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:$PATH

RUN bash -c "source activate ${ENV_NAME} && \
    conda install --yes pytorch torchvision torchaudio cpuonly -c pytorch && \
    # conda env update -f requirements.yaml \
    # conda install --yes jupyter
    # conda clean --all"
    echo 'Done'"

# RUN bash -c "source activate ${ENV_NAME} && \
    # conda install --yes pytorch torchvision torchaudio cpuonly -c pytorch && \
    # conda env update -f requirements.yaml \
    # conda install --yes jupyter
    # conda clean --all"
    # echo 'Done'"



#install pytorch for CPU
# RUN conda install --yes pytorch torchvision torchaudio cpuonly -c pytorch
#install pytorch for GPU
# RUN conda install --yes pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia


# Optional: Install pip packages
# RUN pip install --no-cache-dir \
    # black \
    # pylint \
    # flake8 \
    # pytest

# Optional: Add a non-root user for development
# RUN useradd -m -s /bin/bash devuser && \
    # echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to the non-root user
# USER devuser

# Expose the default JupyterLab port
EXPOSE 8888

# Start the container with a bash shell or JupyterLab
# CMD ["bash"]
CMD ["/bin/bash"]

# Alternatively, you can start JupyterLab by default
# CMD ["jupyter-lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
# jupyter-lab --ip=0.0.0.0 --no-browser --allow-root
