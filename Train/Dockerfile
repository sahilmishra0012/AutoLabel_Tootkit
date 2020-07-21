FROM ubuntu:18.04
RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y git python3-pip
RUN python3 -m pip install --upgrade pip
RUN mkdir imgcls
COPY . imgcls
WORKDIR /imgcls
RUN ls -a
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt
ENV GOOGLE_APPLICATION_CREDENTIALS=edustudent360-c1d02fe3adea.json
#ENTRYPOINT python3 classifier_train.py --data_dir=${DATA_DIR} --model_dir=${MODEL_DIR} --run_eagerly=${RUN_EAGER} --resize=${RESIZE} --multi_worker=${MULTI_WORKER} --epochs=${EPOCHS} --steps_per_epoch=${STEPS_PER_EPOCH}
