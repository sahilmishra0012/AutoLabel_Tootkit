FROM ubuntu:18.04
RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y git python3-pip
RUN python3 -m pip install --upgrade pip
RUN mkdir image_classification
RUN cp . image_classification
RUN ls image_classification -a
RUN pip install -r image_classification/requirements.txt
ENTRYPOINT ["python3", "classifier_train.py "]