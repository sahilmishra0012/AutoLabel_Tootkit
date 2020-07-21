# Image_Classification_Labellerr

## Entry Point Command:

```
python classifier_train.py --data_dir='/path_to_json_file/output2.json' --model_dir='gs://imgcls/Intel' --run_eagerly=true --resize=false --multi_worker=true --epochs=50 --steps_per_epoch=100
```

## GCR Docker Image Pull Command:

```
docker pull gcr.io/aaria-263910/imgclsfinaltry:v1
```