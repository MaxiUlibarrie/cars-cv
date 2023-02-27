# Getting arguments
while getopts e:b:v:w:y:i: flag
do
    case "${flag}" in
        v) version_model=${OPTARG};;
        e) epochs=${OPTARG};;
        b) batch_size=${OPTARG};;
        w) workers=${OPTARG};;
        y) yolo_weights=${OPTARG};;
        i) image_size=${OPTARG};;
    esac
done

case $yolo_weights in
    yolov5n) weights="yolov5n.pt";;
    yolov5s) weights="yolov5s.pt";;
    yolov5m) weights="yolov5m.pt";;
    yolov5l) weights="yolov5l.pt";;
    yolov5x) weights="yolov5x.pt";;
esac

echo "### Creating folder for training ###"
mkdir -p $YOLOV5_PATH/data/street_data

echo "### Copying training data to yolov5 ###"
cp -a $TRAINING_DATA_PATH/. $YOLOV5_PATH/data/street_data

echo "### Training OD model ###"
python $YOLOV5_PATH/train.py --weights $weights \
    --data $YOLOV5_PATH/data/street_data/data.yml \
    --name cars-od \
    --workers $workers \
    --epochs $epochs \
    --batch-size $batch_size \
    --img $image_size \
    --cache

echo "### Creating folder for new version ###"
mkdir -p $NEW_MODELS_PATH/V$version_model

echo "### Saving OD model results ###"
cp -a yolov5/runs/train/cars-od/. $NEW_MODELS_PATH/V$version_model
