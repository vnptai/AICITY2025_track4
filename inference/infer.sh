#!/bin/bash

cd yolo11/build

cmake ..
make -j

./yolo11_det -d ../yolo11m.engine /images c
