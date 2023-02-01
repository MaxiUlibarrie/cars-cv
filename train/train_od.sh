# Getting arguments
while getopts e:b:v:w: flag
do
    case "${flag}" in
        v) version_model=${OPTARG};;
        e) epochs=${OPTARG};;
        b) batch_size=${OPTARG};;
        w) workers=${OPTARG};;
    esac
done

echo "### Creating dir for yolov5 ###"
mkdir yolov5

echo "### Cloning from https://github.com/ultralytics/yolov5 ###"
git clone https://github.com/ultralytics/yolov5.git yolov5/

echo "Installing yolov5 requirements"
pip install -r yolov5/requirements.txt

echo "### Creating folder for training ###"
mkdir yolov5/data/street_data

echo "### Copying training data to yolov5 ###"
cp -r $PATH_TRAINING_DATA yolov5/data/street_data

echo "### Training OD model ###"
python yolov5/train.py --weights yolov5s.pt --data yolov5/data/street_data/data/data.yml --name cars-od --workers $workers --epochs $epochs --batch-size $batch_size

echo "### Creating folder for new version ###"
mkdir $PATH_MODELS/V$version_model

echo "### Saving OD model results ###"
cp -r yolov5/runs/train/cars-od $PATH_MODELS/V$version_model
