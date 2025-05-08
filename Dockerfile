FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libsndfile1 (linux soundfile package)
RUN apt-get clean \
    && apt-get update \ 
    && apt-get install -y gcc g++ libsndfile1 ffmpeg sox wget git \
    && rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
RUN python3 -m pip install --upgrade --no-cache-dir pip wheel \ 
    && python3 -m pip install --no-cache-dir Cython==3.0.6

ADD requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /adpt-test
ENTRYPOINT ["bash"]
