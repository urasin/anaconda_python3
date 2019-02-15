FROM continuumio/anaconda3

ENV LANG C.UTF-8
ENV APP_HOME /scripts
RUN mkdir $APP_HOME
WORKDIR $APP_HOME
ADD . $APP_HOME
RUN pip install -r requirements.txt