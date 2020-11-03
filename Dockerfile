FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y python3 python3-pip
RUN pip3 install torch 
RUN pip3 install -U pip
RUN pip3 install transformers==3.4.0
RUN pip3 install datasets=1.1.2
RUN pip3 install pandas
RUN pip3 install tqdm emoji pythainlp

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
