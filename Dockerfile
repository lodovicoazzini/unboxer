FROM python:3.8

RUN mkdir -p "workspace"
COPY config ./config
COPY feature_map ./feature_map
COPY in ./in
COPY logs ./logs
COPY out ./out
COPY steps ./steps
COPY utils ./utils
COPY requirements.txt ./

RUN pip3 install --upgrade pip
RUN apt-get update
RUN apt-get install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0
RUN apt-get install -y libcairo2-dev pkg-config
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN pip3 install -r ./requirements.txt