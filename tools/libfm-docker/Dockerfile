FROM ubuntu:16.10

MAINTAINER Max Pagels <max.pagels@sc5.io>

RUN apt-get update
RUN apt-get install -y \
    wget \
    make \
    g++

WORKDIR /

ENV LIBFM_VERSION 1.42
RUN wget http://www.libfm.org/libfm-${LIBFM_VERSION}.src.tar.gz
RUN tar -xzvf libfm-${LIBFM_VERSION}.src.tar.gz

WORKDIR libfm-${LIBFM_VERSION}.src/

RUN make all
