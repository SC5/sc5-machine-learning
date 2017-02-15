FROM centos:7

MAINTAINER Max Pagels <max.pagels@sc5.io>

RUN yum update -y && yum install -y \
  epel-release \
  gcc

RUN localedef -c -f UTF-8 -i en_US en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8

RUN yum install -y \
  net-tools \
  git \
  cmake \
  libtool \
  boost \
  boost-devel \
  zlib-devel \
  which

RUN yum group install -y "Development Tools"
RUN git clone https://github.com/JohnLangford/vowpal_wabbit.git
RUN cd vowpal_wabbit/
WORKDIR /vowpal_wabbit
RUN sh autogen.sh
RUN ./configure
RUN make
RUN make install

WORKDIR /

EXPOSE 26542

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
