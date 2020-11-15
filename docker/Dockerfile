ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.0 8.1"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/jms0923/mmdetection.git /aichallenge
WORKDIR /aichallenge
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install pillow
RUN pip install opencv-python
RUN pip install --no-cache-dir -e .

COPY epoch_1.pth /aichallenge/epoch.pth
COPY Chellange_detectors_cascade_rcnn_r50_1x_coco_WorkFestival_4_X3.py /aichallenge/
