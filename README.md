# Prevent Collision System
This is a project to prevent vehicles collisions to people that cross in the middle of the street.

![Alt text](/resources/person_in_danger.gif)

### Require:
* Docker
* Docker Compose

### CI/CD:
These are the principal pipelines for this project (all of them managed using Docker Compose):
* TRAINING
* DETECTION

# TRAINING
Pipeline to automate training OD model process. 

## Build Train Image

> `docker-compose build train`

## Run training model

### Parameters:

#### Require:
* **--version-model**: version of the new model to train

#### Not Require:
* **--epochs**: number of epochs
* **--batch-size**: batch size for training (default: 4)
* **--workers**: number of workers (default: 4)
* **--yolo-weights**: YOLOv5 weights (default: yolov5s, options: ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"])

Example:
> `docker-compose run train --version-model 1 --epochs 3 --batch-size 4 --workers 2 --yolo-weights yolov5s`

# DETECTION
Pipeline to test OD and PC models. It generates a video with the detections found in the original video put in `pipelines/detection/video`. The final video is going to be stored in `pipelines/detection/outputs` after executing the pipeline.

## Build Detection Image

> `docker-compose build detection`

## Run detection pipeline

### Parameters:

#### Require:
* **--weights**: version of the OD model to test.

#### Not Require:
* **--width**: width of images (default: 640).
* **--height**: height of images (default: 480).
* **--conf-thres**: Confidence Threshold used for OD model (default: 0.3).
* **--iop-thres**: Intersection Over Person threshold used to find possible collisions between people and vehicles (default: 0.5).
* **--format-video**: Format of video input and output (default: 'mp4').
* **--video-codec**:  Video codec used for generating video output (default: 'mp4v').

Example:
> `docker-compose run detection --weights V03 --conf-thres 0.4 --iop-thres 0.6`

# Config file
Also in the configuration file it is posible to change these default values (`common/config.json`)

# Intersection Over Person (IOP)
Calculate the percentage of the person (blue box) over the collision zone (yellow polygon). When the IOP is over the IOP threshold the bounding box of the person turns red. This means that the person is in danger because is inside the collision zone in front fo the vehicle. 

![Alt text](/resources/iop.png)
