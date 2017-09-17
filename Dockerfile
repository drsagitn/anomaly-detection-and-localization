FROM nvidia/cuda:8.0-cudnn6-devel

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    chmod -R a+r /opt/conda

RUN apt-get install -y curl grep sed dpkg && \
    apt-get install -y wget git libhdf5-dev g++ graphviz openmpi-bin && \
    apt-get install -y build-essential cmake pkg-config && \
    apt-get install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev && \
    apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev && \
    apt-get install -y libxvidcore-dev libx264-dev && \
    apt-get install -y libgtk-3-dev && \
    apt-get install -y libatlas-base-dev gfortran && \
    apt-get install -y libav-tools && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH

# Python + OpenCV + Tensorflow + Keras
ARG python_version=3.5

RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    pip install tensorflow-gpu && \
    pip install sk-video && \
    pip install tqdm coloredlogs && \
    pip install opencv-contrib-python && \
    conda install -y Pillow scikit-learn scikit-image graphviz pydot notebook pandas matplotlib mkl nose pyyaml six h5py && \
    pip install keras && \
    conda clean -yt

ENV PYTHONPATH='/src/:$PYTHONPATH'

WORKDIR /share
