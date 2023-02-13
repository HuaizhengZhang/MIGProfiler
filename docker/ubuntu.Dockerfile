FROM nvcr.io/nvidia/pytorch:22.10-py3
LABEL maintainer="Li Yuanming <yuanmingleee@gmail.com>"
ENV PYTHONPATH /workspace

WORKDIR /workspace

# Install required packages
RUN pip install --no-cache-dir transformers -i https://pypi.mirrors.ustc.edu.cn/simple

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple

# Build migperf
RUN rm -r .
COPY . .
RUN pip install --no-cache-dir .

ENTRYPOINT ["/bin/bash"]