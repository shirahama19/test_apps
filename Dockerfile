FROM ubuntu
WORKDIR /test_apps

LABEL maintainer Koki Shirahama<kshirahama@netprotections.co.jp>

RUN apt-get update
RUN apt-get insall python3-pip -y

RUN pip3 insall numpy pandas sklearn seaborn
RUN pip3 insall chainer Flask

EXPOSE 5000
CMD ["python3", "server.py"]
