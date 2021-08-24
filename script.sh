#!/bin/zsh

MAIN_PATH=/Users/erichuang/Desktop/Eye
MEDIA_PATH=/mediapipe-0.8.2
MATER_PATH=/material

cd ${MAIN_PATH}${MEDIA_PATH}
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/iris_tracking:iris_depth_from_image_desktop

for file in $(ls ${MAIN_PATH}${MATER_PATH}/*.png)
do
    # ffmpeg -i $file ${file%.*}.jpg
    cd ${MAIN_PATH}${MEDIA_PATH}
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_depth_from_image_desktop \
  --input_image_path=${file} --output_image_path=${file%.*}_tracking.jpg
    
    cd ${MAIN_PATH}
    python demo.py ${file}
    
    rm ${file%.*}_tracking.jpg
done
