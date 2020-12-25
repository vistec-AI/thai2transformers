FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y python3 python3-pip
RUN pip3 install torch 
RUN pip3 install -U pip
RUN apt-get install -y git
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && \
    pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN pip3 install pandas
RUN pip3 install tqdm emoji pythainlp==2.2.4
RUN pip3 install transformers==3.4.0
RUN git clone https://github.com/huggingface/datasets.git && cd datasets && \
    pip3 install .
RUN pip3 install tensorboard==2.3.0
RUN pip3 install sefr_cut

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
