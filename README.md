# Prevent Collision System
This is a project to prevent cars collisions to people that cross in the middle of the street.

### Require:
* Docker
* Docker Compose

### CI/CD:
There are two principal processes for this project (both managed using Docker Compose):
* TRAINING
* BACKEND

# TRAINING

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

### Config file
Also in the configuration file it is posible to change these default values (`common/config.json`)

# BACKEND

## Build Image 

> `docker-compose build backend`

## Getting service up

> `docker-compose up backend`

## Endpoints

...