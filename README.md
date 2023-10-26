# FisheyeVision
A Vehicle Movement Counting System for Intersections Monitored by A Fisheye Camera

Implementation of paper - [Fisheye Vision: A Novel AI-Powered Vehicle Movement Counting System for Intersections]

# Installation
``` shell
cd code_directory
pip install -r requirements.txt
```

Download the preferred [YOLOv7 ](https://github.com/WongKinYiu/yolov7) weights to the code directory.

[`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) [`yolov7x.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) [`yolov7-w6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) [`yolov7-e6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) [`yolov7-d6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) [`yolov7-e6e.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)

# Configuration
Run config.py and select one of the images from your dataset to generate a cfg.txt file and define zones for the intersection.

Open Counter/cfg/cfg.txt and define the number of lanes for each street.

## Training
On video:
``` shell
python DTC.py --weights yolov7-e6e.pt --conf 0.25 --img-size 640 --source inference/yourvideo.mp4 --LearnPatterns --TracksPerLane 50
```
On image:
``` shell
python DTC.py --weights yolov7-e6e.pt --conf 0.25 --img-size 640 --source inference/images_folder --LearnPatterns --TracksPerLane 50
```

## Inference
On video:
``` shell
python DTC.py --weights yolov7.pt --conf 0.25 --view-img --img-size 640 --source yourvideo.mp4
```
On image:
``` shell
python DTC.py --weights yolov7.pt --conf 0.25 --view-img --img-size 640 --source inference/images_folder
```
