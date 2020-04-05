# Stable tensorflow base image
FROM tensorflow/tensorflow:1.15.0-gpu-py3

# Get the tensorflow models research directory, and move it into tensorflow
# source folder to match recommendation of installation

# Install wget (to make life easier below) and editors (to allow people to edit
# the files inside the container)
RUN apt-get update --fix-missing
RUN apt-get install -y git wget vim emacs nano

RUN mkdir tensorflow

RUN git clone --depth 1 https://github.com/tensorflow/models.git && \
    mv models /tensorflow/models

# Install the Tensorflow Object Detection API from here
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

# Install object detection api dependencies
# Note: requires time zone information: https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y protobuf-compiler python-pil python-lxml python-tk && \
    pip install Cython && \
    pip install contextlib2 && \
    pip install jupyter && \
    pip install matplotlib

# Install pycocoapi
RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    make -j8 && \
    cp -r pycocotools /tensorflow/models/research && \
    cd ../../ && \
    rm -rf cocoapi

# Get protoc 3.0.0, rather than the old version already in the container
RUN curl -OL "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip" && \
    unzip protoc-3.0.0-linux-x86_64.zip -d proto3 && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-3.0.0-linux-x86_64.zip

# Run protoc on the object detection repo
RUN cd /tensorflow/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

# Set the PYTHONPATH to finish installing the API
ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim

# Grab various data files which are used throughout the demo: dataset,
# pretrained model, and pretrained TensorFlow Lite model. Install these all in
# the same directories as recommended by the blog post.


# # Pretrained model
# # This one doesn't need its own directory, since it comes in a folder.
# RUN cd /tmp && \
#     curl -O "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz" && \
#     tar xzf ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz && \
#     rm ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz
#
# # Trained TensorFlow Lite model. This should get replaced by one generated from
# # export_tflite_ssd_graph.py when that command is called.
# RUN cd /tmp && \
#     curl -L -o tflite.zip \
#     https://storage.googleapis.com/download.tensorflow.org/models/tflite/frozengraphs_ssd_mobilenet_v1_0.75_quant_pets_2018_06_29.zip && \
#     unzip tflite.zip -d tflite && \
#     rm tflite.zip

# TODO: doe dit eerder
RUN pip install pillow

WORKDIR /tensorflow

# Create custom object detection
RUN mkdir -p workspace/training_demo

WORKDIR /tensorflow/workspace/training_demo
RUN mkdir -p annotations pre-trained-model training images/train images/test