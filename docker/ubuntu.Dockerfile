FROM nvcr.io/nvidia/pytorch:22.10-py3
LABEL maintainer="Li Yuanming <yuanmingleee@gmail.com>"
ENV PYTHONPATH /workspace

WORKDIR /workspace

# Install required packages
RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN conda install -c conda-forge opencv && pip install --no-cache-dir transformers \
    && conda clean -ya

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build migperf
COPY migperf migperf
RUN pip install --no-cache-dir .

ENTRYPOINT ["/bin/bash"]