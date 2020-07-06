# Pose Data Extractor
Using Pose Detection to estimate the attention span of people in an online lecture session. The project uses OpenPose to detect keypoints from the webcam. The video data obtained is split into frames and the data retreived is solely the OpenPose figures, on a black background. Addiional, keypoint coordinate data for each of these frames is maintained and using the distance and angle between the points. This data can then be used to extrapolate essential factors that affect a person's attentivity.

## Updated repository
This repository has been further updated to use the extracted data, and use it to perform attention detection with the help of images as well as keypoint **.csv** data.
_Currently under progress. Full project will be updated soon._

## Download the OpenPose model from here
https://drive.google.com/file/d/1Fqyn90yIUOxgT1PC2QHwkh_CwvWSFuXy/view?usp=sharing

Add the **model.h5** file to the working directory
