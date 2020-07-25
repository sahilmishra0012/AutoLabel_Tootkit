#his is an official Python runtime, used as the parent image
FROM python:3.7-slim

# set the working directory in the container to /app
WORKDIR /app

# add the current directory to the container as /app
ADD . /app

ENV GCLOUD_PROJECT='edustudent360'

#execute everyone's favorite pip command, pip install -r
ENV GOOGLE_APPLICATION_CREDENTIALS=edustudent360-c1d02fe3adea.json

RUN pip install -r requirements.txt

# execute the Flask app
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app