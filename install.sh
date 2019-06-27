#!/bin/sh
sudo locale-gen en_US.UTF-8
sudo locale-gen ru_RU.UTF-8
sudo dpkg-reconfigure locales

sudo add-apt-repository ppa:chris-lea/node.js -y
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python3 python3-setuptools python3-dev python3-lxml libpq-dev  supervisor  rabbitmq-server  memcached  git  mercurial build-essential libjpeg8 libjpeg-dev  libfreetype6  libfreetype6-dev  zlib1g-dev  libxml2-dev libxslt1-dev unzip
echo oracle-java7-installer shared/accepted-oracle-license-v1-1 select true | sudo /usr/bin/debconf-set-selections
sudo apt-get install -y oracle-java7-installer
sudo apt-get purge python3-pip
sudo easy_install3 pip
sudo apt-get install -y nodejs  ruby-full
sudo apt-get install build-essential cmake unzip pkg-config
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python3-tk
