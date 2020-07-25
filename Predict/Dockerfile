FROM ubuntu:18.04
RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y git python3-pip
RUN python3 -m pip install --upgrade pip
RUN mkdir prediction
COPY . prediction
RUN ls prediction -a
WORKDIR /prediction
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt
RUN pip install flask
RUN pip install Flask gunicorn
ENV PORT 8080
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app